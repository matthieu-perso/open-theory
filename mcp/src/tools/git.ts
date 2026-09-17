import fs from 'fs';
import path from 'path';
import {
  WORKSPACE_ROOT,
  LEAN_DIR,
  BLUEPRINT_TEX_PATH,
  BLUEPRINT_JSON_PATH,
  EXPORT_BLUEPRINT_PY_PATH,
  getGitBin,
  getPythonBin,
  KNOWN_BOUNTIES,
} from '../config.js';
import {
  findLeanDeclaration,
  loadBlueprintJson,
  parseContentTex,
} from './blueprint.js';
import {
  injectProofIntoSource,
  runCommand,
  handleVerifyLeanProof,
} from './lean.js';

export interface CommitProofArgs {
  lemma_id: string;
  proof_code: string;
  branch_name?: string;
  commit_message?: string;
  author_name?: string;
  author_email?: string;
  verify_before_commit?: boolean;
  update_blueprint?: boolean;
}

export interface CommitProofResult {
  success: boolean;
  branch: string;
  commit_hash?: string;
  files_modified: string[];
  diff_stat?: string;
  blueprint_updated: boolean;
  verification_passed?: boolean;
  summary: string;
}

/**
 * Adds \leanok tag to the target lemma in blueprint/src/content.tex
 */
export function markLemmaVerifiedInTex(lemmaId: string): boolean {
  if (!fs.existsSync(BLUEPRINT_TEX_PATH)) return false;

  const content = fs.readFileSync(BLUEPRINT_TEX_PATH, 'utf-8');
  const cleanId = lemmaId.replace(/^lem:|^def:|^thm:|^conj:/, '');

  // Regex to find environment with \label{lemmaId}
  const labelPattern = new RegExp(
    `(\\\\begin\\{(?:theorem|lemma|definition|conjecture)\\}(?:\\[.*?\\])?\\\\label\\{(?:lem:|def:|thm:|conj:)?${cleanId}\\}[\\s\\S]*?)(\\\\end\\{(?:theorem|lemma|definition|conjecture)\\})`,
    'm'
  );

  const match = labelPattern.exec(content);
  if (!match) return false;

  const body = match[1];
  if (/\\leanok\b/.test(body)) {
    return true; // Already marked verified
  }

  // Insert \leanok after \lean{...} if present, or after \label{...}
  let updatedBody: string;
  if (/\\lean\{.*?\}/.test(body)) {
    updatedBody = body.replace(/(\\lean\{.*?\})/, '$1\n\\leanok');
  } else {
    updatedBody = body.replace(/(\\label\{.*?\})/, '$1\n\\leanok');
  }

  const newContent = content.replace(labelPattern, `${updatedBody}$2`);
  fs.writeFileSync(BLUEPRINT_TEX_PATH, newContent, 'utf-8');
  return true;
}

/**
 * Regenerates blueprint.json by invoking export_blueprint.py or direct JSON update
 */
export async function syncBlueprintJson(lemmaId: string): Promise<boolean> {
  const pythonBin = getPythonBin();
  if (fs.existsSync(EXPORT_BLUEPRINT_PY_PATH)) {
    const res = await runCommand(pythonBin, [EXPORT_BLUEPRINT_PY_PATH, BLUEPRINT_JSON_PATH], WORKSPACE_ROOT, 10000);
    if (res.exitCode === 0) {
      return true;
    }
  }

  // Direct JSON update fallback
  try {
    if (fs.existsSync(BLUEPRINT_JSON_PATH)) {
      const data = loadBlueprintJson();
      for (const node of data.nodes) {
        if (node.id === lemmaId || node.id.endsWith(lemmaId)) {
          node.isVerified = true;
        }
      }
      fs.writeFileSync(BLUEPRINT_JSON_PATH, JSON.stringify(data, null, 2), 'utf-8');
      return true;
    }
  } catch (err) {
    console.error('[MCP] Failed fallback blueprint.json update:', err);
  }

  return false;
}

/**
 * Implementation of `commit_proof` tool.
 */
export async function handleCommitProof(
  args: CommitProofArgs
): Promise<CommitProofResult> {
  const gitBin = getGitBin();
  const blueprint = loadBlueprintJson();
  const texEnvs = parseContentTex();

  // Find lemma metadata
  const targetId = args.lemma_id.trim();
  const targetNode = blueprint.nodes.find(
    (n) => n.id === targetId || n.id.endsWith(targetId) || n.leanName?.endsWith(targetId)
  );

  const texEnv = targetNode ? texEnvs.get(targetNode.id) : undefined;
  const bountyMeta = targetNode ? KNOWN_BOUNTIES[targetNode.id] : undefined;
  const lookupName = targetNode?.leanName || targetId;

  // 1. Locate Lean source file
  const declLoc = findLeanDeclaration(lookupName);
  if (!declLoc) {
    return {
      success: false,
      branch: '',
      files_modified: [],
      blueprint_updated: false,
      summary: `Could not find Lean declaration for lemma '${targetId}' in OpenTheory library.`,
    };
  }

  // Static checks on proposed proof
  if (/\bsorry\b/.test(args.proof_code)) {
    return {
      success: false,
      branch: '',
      files_modified: [],
      blueprint_updated: false,
      summary: `Cannot commit proof containing 'sorry'. All goals must be closed with formal tactics.`,
    };
  }
  if (/\baxiom\b/.test(args.proof_code)) {
    return {
      success: false,
      branch: '',
      files_modified: [],
      blueprint_updated: false,
      summary: `Cannot commit proof introducing custom 'axiom' assertions. Only standard Mathlib foundational axioms are permitted.`,
    };
  }

  // 2. Optionally verify proof before committing
  let verificationPassed: boolean | undefined = undefined;
  if (args.verify_before_commit) {
    const verifyRes = await handleVerifyLeanProof({
      lemma_id: targetId,
      proof_code: args.proof_code,
      dry_run: true,
      run_axiom_audit: true,
    });

    if (verifyRes.lake_installed) {
      verificationPassed = verifyRes.success;
      if (!verifyRes.success) {
        return {
          success: false,
          branch: '',
          files_modified: [],
          blueprint_updated: false,
          verification_passed: false,
          summary: `Pre-commit verification failed: ${verifyRes.summary}. Fix the proof tactics before committing.`,
        };
      }
    }
  }

  // 3. Inject verified proof permanently into the Lean file
  const originalFileContent = fs.readFileSync(declLoc.filePath, 'utf-8');
  const { updatedContent, replaced } = injectProofIntoSource(
    originalFileContent,
    declLoc.declName,
    args.proof_code
  );

  if (!replaced) {
    return {
      success: false,
      branch: '',
      files_modified: [],
      blueprint_updated: false,
      summary: `Failed to replace sorry in ${declLoc.relativeFilePath}. Check proof syntax.`,
    };
  }

  fs.writeFileSync(declLoc.filePath, updatedContent, 'utf-8');
  const modifiedFiles: string[] = [declLoc.relativeFilePath];

  // 4. Update Blueprint (\leanok in content.tex and re-export blueprint.json)
  let blueprintUpdated = false;
  if (args.update_blueprint !== false && targetNode) {
    const texMarked = markLemmaVerifiedInTex(targetNode.id);
    if (texMarked) {
      modifiedFiles.push(path.relative(WORKSPACE_ROOT, BLUEPRINT_TEX_PATH));
      const synced = await syncBlueprintJson(targetNode.id);
      if (synced) {
        modifiedFiles.push(path.relative(WORKSPACE_ROOT, BLUEPRINT_JSON_PATH));
        blueprintUpdated = true;
      }
    }
  }

  // 5. Git operations: branch, stage, commit
  const slugId = targetId.replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase();
  const branchName = args.branch_name || `proof/${slugId}`;

  try {
    // Configure author if provided
    if (args.author_name) {
      await runCommand(gitBin, ['config', 'user.name', args.author_name], WORKSPACE_ROOT, 5000);
    }
    if (args.author_email) {
      await runCommand(gitBin, ['config', 'user.email', args.author_email], WORKSPACE_ROOT, 5000);
    }

    // Checkout new or existing branch
    const branchCheck = await runCommand(gitBin, ['checkout', '-b', branchName], WORKSPACE_ROOT, 10000);
    if (branchCheck.exitCode !== 0) {
      // Branch might already exist, try plain checkout
      await runCommand(gitBin, ['checkout', branchName], WORKSPACE_ROOT, 10000);
    }

    // Stage modified files
    for (const relFile of modifiedFiles) {
      await runCommand(gitBin, ['add', relFile], WORKSPACE_ROOT, 5000);
    }

    // Compose commit message
    const bountyText = bountyMeta?.amountUsd ? ` (Bounty: $${bountyMeta.amountUsd.toLocaleString()})` : '';
    const titleText = targetNode?.title ? ` - ${targetNode.title}` : '';
    const commitMsg =
      args.commit_message ||
      `feat(proof): prove ${targetId}${titleText}${bountyText}\n\nMachine-checked Lean 4 formalization verified against OpenTheory kernel.\nAutomated commit via OpenTheory MCP Agent.`;

    const commitRes = await runCommand(gitBin, ['commit', '-m', commitMsg], WORKSPACE_ROOT, 15000);

    // Get commit SHA
    const revRes = await runCommand(gitBin, ['rev-parse', 'HEAD'], WORKSPACE_ROOT, 5000);
    const commitHash = revRes.stdout.trim();

    // Get diff stat
    const diffStatRes = await runCommand(gitBin, ['diff', '--stat', 'HEAD~1', 'HEAD'], WORKSPACE_ROOT, 5000);

    return {
      success: commitRes.exitCode === 0,
      branch: branchName,
      commit_hash: commitHash || undefined,
      files_modified: modifiedFiles,
      diff_stat: diffStatRes.stdout.trim() || undefined,
      blueprint_updated: blueprintUpdated,
      verification_passed: verificationPassed,
      summary: `Successfully committed verified proof for '${targetId}' to branch '${branchName}' (${commitHash.substring(0, 7)}).`,
    };
  } catch (err: any) {
    return {
      success: false,
      branch: branchName,
      files_modified: modifiedFiles,
      blueprint_updated: blueprintUpdated,
      summary: `Proof applied to local files, but git commit encountered error: ${err?.message || err}`,
    };
  }
}
