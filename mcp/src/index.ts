#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import {
  handleGetOpenBounties,
  handleGetLemmaContext,
} from './tools/blueprint.js';
import {
  handleVerifyLeanProof,
} from './tools/lean.js';
import {
  handleCommitProof,
} from './tools/git.js';
import {
  WORKSPACE_ROOT,
  LEAN_DIR,
  BLUEPRINT_JSON_PATH,
  getLakeBin,
} from './config.js';

// Initialize the OpenTheory MCP Server
const server = new McpServer({
  name: 'open-theory-core',
  version: '0.1.0',
});

// Tool 1: get_open_bounties
server.tool(
  'get_open_bounties',
  'Lists all lemmas with status "unproven" or "sorry", filtered by bounty value (USD), declaration kind, or dependency readiness.',
  {
    min_bounty: z
      .number()
      .optional()
      .describe('Minimum bounty reward in USD (e.g. 1000)'),
    max_bounty: z
      .number()
      .optional()
      .describe('Maximum bounty reward in USD (e.g. 5000)'),
    kind: z
      .enum(['all', 'lemma', 'theorem', 'conjecture'])
      .optional()
      .describe('Filter by declaration kind (lemma, theorem, conjecture)'),
    status: z
      .enum(['all', 'unproven', 'sorry'])
      .optional()
      .describe("Filter by status: 'sorry' (formalized with sorry), 'unproven' (unformalized), or 'all' open"),
    only_ready_to_prove: z
      .boolean()
      .optional()
      .describe('If true, only returns lemmas whose prerequisite dependencies are already verified'),
  },
  async (args) => {
    try {
      const result = handleGetOpenBounties(args);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (err: any) {
      return {
        isError: true,
        content: [
          {
            type: 'text',
            text: `Error fetching open bounties: ${err?.message || err}`,
          },
        ],
      };
    }
  }
);

// Tool 2: get_lemma_context
server.tool(
  'get_lemma_context',
  'Returns the mathematical LaTeX statement, informal proof sketch, current Lean 4 declaration with sorry, and proven predecessor lemmas with their formal definitions.',
  {
    lemma_id: z
      .string()
      .describe("Lemma or conjecture identifier, e.g. 'lem:spectral_bias_mup', 'LEM-2.1', or 'spectral_bias_mup_alignment'"),
  },
  async (args) => {
    try {
      const result = handleGetLemmaContext(args);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (err: any) {
      return {
        isError: true,
        content: [
          {
            type: 'text',
            text: `Error getting context for '${args.lemma_id}': ${err?.message || err}`,
          },
        ],
      };
    }
  }
);

// Tool 3: verify_lean_proof
server.tool(
  'verify_lean_proof',
  'Tests Lean 4 code against the kernel: checks syntax, open goals (sorry), compiler errors/warnings, and executes scripts/audit_axioms.lean to reject unauthorized axioms.',
  {
    lemma_id: z
      .string()
      .optional()
      .describe("Target lemma ID or Lean name (e.g. 'lem:spectral_bias_mup' or 'spectral_bias_mup_alignment')"),
    proof_code: z
      .string()
      .optional()
      .describe('Lean 4 proof tactics or term to test replacing the sorry placeholder (e.g. "use gram_eigenvalue\\nconstructor\\n...")'),
    file_path: z
      .string()
      .optional()
      .describe('Specific Lean file to verify (relative to lean workspace root)'),
    run_axiom_audit: z
      .boolean()
      .optional()
      .describe('Whether to run scripts/audit_axioms.lean (default: true)'),
    dry_run: z
      .boolean()
      .optional()
      .describe('If true (default), automatically restores original files after verification so testing is non-destructive'),
  },
  async (args) => {
    try {
      const result = await handleVerifyLeanProof(args);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (err: any) {
      return {
        isError: true,
        content: [
          {
            type: 'text',
            text: `Lean proof verification failed with exception: ${err?.message || err}`,
          },
        ],
      };
    }
  }
);

// Tool 4: commit_proof
server.tool(
  'commit_proof',
  'Replaces sorry with the verified proof in the corresponding .lean file, marks \\leanok in blueprint/src/content.tex, refreshes blueprint.json, and commits to a Git branch.',
  {
    lemma_id: z
      .string()
      .describe("Target lemma ID or Lean name, e.g. 'lem:spectral_bias_mup'"),
    proof_code: z
      .string()
      .describe('Verified Lean 4 proof tactics or term that replaces sorry'),
    branch_name: z
      .string()
      .optional()
      .describe("Git branch name (defaults to 'proof/<lemma_id_slug>')"),
    commit_message: z
      .string()
      .optional()
      .describe('Git commit message (auto-composed if omitted)'),
    author_name: z
      .string()
      .optional()
      .describe('Git author name'),
    author_email: z
      .string()
      .optional()
      .describe('Git author email'),
    verify_before_commit: z
      .boolean()
      .optional()
      .describe('Runs verify_lean_proof before committing to ensure the proof checks out (default: true)'),
    update_blueprint: z
      .boolean()
      .optional()
      .describe('Automatically adds \\leanok to blueprint/src/content.tex and syncs blueprint.json (default: true)'),
  },
  async (args) => {
    try {
      const result = await handleCommitProof(args);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (err: any) {
      return {
        isError: true,
        content: [
          {
            type: 'text',
            text: `Failed to commit proof for '${args.lemma_id}': ${err?.message || err}`,
          },
        ],
      };
    }
  }
);

// Main entrypoint
async function main() {
  const lakeBin = getLakeBin();
  process.stderr.write(
    `[OpenTheory MCP Server] Starting... (Workspace: ${WORKSPACE_ROOT}, Lake: ${lakeBin || 'not found in PATH'})\n`
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);

  process.stderr.write('[OpenTheory MCP Server] Connected to stdio transport successfully.\n');
}

main().catch((err) => {
  process.stderr.write(`[OpenTheory MCP Server] Fatal startup error: ${err?.stack || err}\n`);
  process.exit(1);
});
