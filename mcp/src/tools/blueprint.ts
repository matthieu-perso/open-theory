import fs from 'fs';
import path from 'path';
import {
  BLUEPRINT_JSON_PATH,
  BLUEPRINT_TEX_PATH,
  OPENTHEORY_LEAN_DIR,
  LEAN_DIR,
  KNOWN_BOUNTIES,
  BountyMetadata,
} from '../config.js';

export interface BlueprintNode {
  id: string;
  kind: string;
  title: string;
  leanName?: string | null;
  isVerified: boolean;
  dependencies: string[];
}

export interface BlueprintData {
  generatedAt?: string;
  toolchain?: string;
  nodesCount: number;
  nodes: BlueprintNode[];
}

export interface ParsedTexEnv {
  id: string;
  kind: string;
  title: string;
  leanName?: string;
  isLeanok: boolean;
  dependencies: string[];
  statement: string;
  proofSketch?: string;
  bountyUsd?: number;
}

export interface LeanDeclLocation {
  filePath: string;
  relativeFilePath: string;
  lineNumber: number;
  declName: string;
  declKind: string;
  fullCode: string;
  hasSorry: boolean;
}

/**
 * Loads blueprint.json data, falling back to an empty set if not found.
 */
export function loadBlueprintJson(): BlueprintData {
  if (!fs.existsSync(BLUEPRINT_JSON_PATH)) {
    return { nodesCount: 0, nodes: [] };
  }
  try {
    const raw = fs.readFileSync(BLUEPRINT_JSON_PATH, 'utf-8');
    return JSON.parse(raw) as BlueprintData;
  } catch (err) {
    console.error(`[MCP] Failed to parse ${BLUEPRINT_JSON_PATH}:`, err);
    return { nodesCount: 0, nodes: [] };
  }
}

/**
 * Parses blueprint/src/content.tex into structured environments and proof sketches.
 */
export function parseContentTex(): Map<string, ParsedTexEnv> {
  const map = new Map<string, ParsedTexEnv>();
  if (!fs.existsSync(BLUEPRINT_TEX_PATH)) {
    return map;
  }

  const text = fs.readFileSync(BLUEPRINT_TEX_PATH, 'utf-8');

  const envRegex =
    /\\begin\{(theorem|lemma|definition|conjecture|proposition|corollary)\}(?:\[(.*?)\])?\s*\\label\{(.*?)\}([\s\S]*?)\\end\{\1\}/g;
  let match: RegExpExecArray | null;

  while ((match = envRegex.exec(text)) !== null) {
    const kind = match[1];
    const title = match[2] ? match[2].trim() : match[3].trim();
    const id = match[3].trim();
    const rawBody = match[4];

    const leanMatch = /\\lean\{(.*?)\}/.exec(rawBody);
    const leanName = leanMatch ? leanMatch[1].split(',')[0].trim() : undefined;

    const usesMatch = /\\uses\{(.*?)\}/.exec(rawBody);
    const dependencies = usesMatch
      ? usesMatch[1].split(',').map((u) => u.trim()).filter(Boolean)
      : [];

    const statement = rawBody
      .replace(/\\lean\{.*?\}/g, '')
      .replace(/\\leanok/g, '')
      .replace(/\\uses\{.*?\}/g, '')
      .trim();

    const afterEnv = text.slice(match.index + match[0].length);
    const proofMatch = /^\s*\\begin\{proof\}([\s\S]*?)\\end\{proof\}/.exec(afterEnv);
    let proofSketch: string | undefined;
    let bountyUsd: number | undefined;
    let proofOk = false;

    if (proofMatch) {
      proofOk = /\\leanok\b/.test(proofMatch[1]);
      proofSketch = proofMatch[1].replace(/\\leanok/g, '').trim();
      const bountyMatch = /Bounty\s+\\?\$?([0-9,]+)/i.exec(proofSketch);
      if (bountyMatch) {
        bountyUsd = parseInt(bountyMatch[1].replace(/,/g, ''), 10);
      }
    }

    const isLeanok = kind === 'definition' ? /\\leanok\b/.test(rawBody) : proofOk;

    // Fallback to known bounties if not explicitly parsed in LaTeX
    if (!bountyUsd && KNOWN_BOUNTIES[id]) {
      bountyUsd = KNOWN_BOUNTIES[id].amountUsd;
    }

    map.set(id, {
      id,
      kind,
      title,
      leanName,
      isLeanok,
      dependencies,
      statement,
      proofSketch,
      bountyUsd,
    });
  }

  return map;
}

/**
 * Recursively find all .lean files in a directory.
 */
export function getAllLeanFiles(dir: string): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...getAllLeanFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.lean')) {
      results.push(fullPath);
    }
  }
  return results;
}

/**
 * Searches Lean files for a given declaration name or identifier.
 */
export function findLeanDeclaration(targetLeanName: string): LeanDeclLocation | null {
  const shortName = targetLeanName.split('.').pop() || targetLeanName;
  const leanFiles = getAllLeanFiles(LEAN_DIR);

  for (const file of leanFiles) {
    const content = fs.readFileSync(file, 'utf-8');
    const lines = content.split('\n');

    // Look for: theorem|lemma|def|structure|inductive <shortName>
    const declRegex = new RegExp(
      `^\\s*(?:(?:protected|noncomputable|scoped|private)\\s+)*(?:@[^\\n]+\\s+)*(theorem|lemma|def|structure|inductive|class)\\s+(?:[A-Za-z0-9_.]*\\.)?(${shortName})\\b`,
      'm'
    );

    const match = declRegex.exec(content);
    if (match) {
      // Find line number
      const charIndex = match.index;
      const lineNumber = content.substring(0, charIndex).split('\n').length;
      const declKind = match[1];

      // Extract declaration block (up to next top-level decl or end of file)
      const afterMatch = content.slice(charIndex);
      const nextDeclMatch = afterMatch.slice(match[0].length).search(/\n(?=(theorem|lemma|def|structure|inductive|class|end\s+OpenTheory)\b)/);
      const declBlock = nextDeclMatch !== -1
        ? afterMatch.slice(0, match[0].length + nextDeclMatch)
        : afterMatch;

      const hasSorry = /\bsorry\b/.test(declBlock);

      return {
        filePath: file,
        relativeFilePath: path.relative(LEAN_DIR, file),
        lineNumber,
        declName: shortName,
        declKind,
        fullCode: declBlock.trim(),
        hasSorry,
      };
    }
  }

  return null;
}

/**
 * Implementation of `get_open_bounties` tool.
 */
export interface GetOpenBountiesArgs {
  min_bounty?: number;
  max_bounty?: number;
  kind?: 'all' | 'lemma' | 'theorem' | 'conjecture';
  status?: 'all' | 'unproven' | 'sorry';
  only_ready_to_prove?: boolean;
}

export function handleGetOpenBounties(args: GetOpenBountiesArgs = {}) {
  const blueprint = loadBlueprintJson();
  const texEnvs = parseContentTex();
  const results = [];

  for (const node of blueprint.nodes) {
    const texEnv = texEnvs.get(node.id);
    const bountyMeta = KNOWN_BOUNTIES[node.id];
    const bountyUsd = texEnv?.bountyUsd ?? bountyMeta?.amountUsd ?? null;

    // Check Lean source code
    let hasSorry = false;
    let leanLoc: LeanDeclLocation | null = null;
    if (node.leanName) {
      leanLoc = findLeanDeclaration(node.leanName);
      if (leanLoc) {
        hasSorry = leanLoc.hasSorry;
      }
    }

    // Determine status: "sorry" | "unproven" | "verified"
    let status: 'verified' | 'sorry' | 'unproven';
    if (node.isVerified && !hasSorry) {
      status = 'verified';
    } else if (hasSorry) {
      status = 'sorry';
    } else {
      status = 'unproven';
    }

    // Check dependency satisfaction
    const unverifiedDeps: string[] = [];
    for (const depId of node.dependencies) {
      const depNode = blueprint.nodes.find((n) => n.id === depId);
      if (!depNode || !depNode.isVerified) {
        unverifiedDeps.push(depId);
      }
    }
    const dependenciesReady = unverifiedDeps.length === 0;

    // Filter by status (we only return open / unverified by default)
    if (args.status === 'sorry' && status !== 'sorry') continue;
    if (args.status === 'unproven' && status !== 'unproven') continue;
    if (!args.status || args.status === 'all') {
      if (status === 'verified') continue; // Default: list open targets only
    }

    // Filter by bounty
    if (args.min_bounty !== undefined && (bountyUsd === null || bountyUsd < args.min_bounty)) {
      continue;
    }
    if (args.max_bounty !== undefined && (bountyUsd !== null && bountyUsd > args.max_bounty)) {
      continue;
    }

    // Filter by kind
    if (args.kind && args.kind !== 'all' && node.kind.toLowerCase() !== args.kind.toLowerCase()) {
      continue;
    }

    // Filter by ready to prove
    if (args.only_ready_to_prove && !dependenciesReady) {
      continue;
    }

    results.push({
      id: node.id,
      code_id: bountyMeta?.codeId || null,
      title: node.title,
      kind: node.kind,
      status,
      bounty_usd: bountyUsd,
      sponsor: bountyMeta?.sponsor || (bountyUsd ? 'OpenTheory Research Escrow' : null),
      escrow_address: bountyMeta?.escrowAddress || null,
      lean_name: node.leanName || null,
      lean_file: leanLoc ? leanLoc.relativeFilePath : null,
      line_number: leanLoc ? leanLoc.lineNumber : null,
      has_sorry_in_source: hasSorry,
      dependencies: node.dependencies,
      dependencies_ready: dependenciesReady,
      unverified_dependencies: unverifiedDeps,
      latex_statement: texEnv?.statement || null,
      proof_sketch: texEnv?.proofSketch || null,
    });
  }

  // Sort by bounty USD descending
  results.sort((a, b) => (b.bounty_usd || 0) - (a.bounty_usd || 0));

  return {
    count: results.length,
    open_bounties: results,
  };
}

/**
 * Implementation of `get_lemma_context` tool.
 */
export interface GetLemmaContextArgs {
  lemma_id: string;
}

export function handleGetLemmaContext(args: GetLemmaContextArgs) {
  const query = args.lemma_id.trim();
  const blueprint = loadBlueprintJson();
  const texEnvs = parseContentTex();

  // Find target node by ID, codeId, or leanName
  let targetNode: BlueprintNode | undefined = blueprint.nodes.find(
    (n) => n.id.toLowerCase() === query.toLowerCase()
  );

  if (!targetNode) {
    // Try matching leanName
    targetNode = blueprint.nodes.find(
      (n) => n.leanName && (n.leanName.toLowerCase() === query.toLowerCase() || n.leanName.toLowerCase().endsWith(query.toLowerCase()))
    );
  }

  if (!targetNode) {
    // Try matching codeId in KNOWN_BOUNTIES
    for (const [id, meta] of Object.entries(KNOWN_BOUNTIES)) {
      if (meta.codeId && meta.codeId.toLowerCase() === query.toLowerCase()) {
        targetNode = blueprint.nodes.find((n) => n.id === id);
        if (targetNode) break;
      }
    }
  }

  // If still not found, search in content.tex environments
  let texEnv = targetNode ? texEnvs.get(targetNode.id) : undefined;
  if (!targetNode) {
    for (const [id, env] of texEnvs.entries()) {
      if (id.toLowerCase() === query.toLowerCase() || (env.leanName && env.leanName.toLowerCase().endsWith(query.toLowerCase()))) {
        texEnv = env;
        targetNode = {
          id: env.id,
          kind: env.kind,
          title: env.title,
          leanName: env.leanName,
          isVerified: env.isLeanok,
          dependencies: env.dependencies,
        };
        break;
      }
    }
  }

  if (!targetNode) {
    throw new Error(
      `Lemma or conjecture '${query}' not found in blueprint. Run get_open_bounties to list valid lemma IDs.`
    );
  }

  texEnv = texEnv || texEnvs.get(targetNode.id);
  const bountyMeta = KNOWN_BOUNTIES[targetNode.id];
  const bountyUsd = texEnv?.bountyUsd ?? bountyMeta?.amountUsd ?? null;

  // Find target declaration in Lean source
  let targetLeanLoc: LeanDeclLocation | null = null;
  if (targetNode.leanName) {
    targetLeanLoc = findLeanDeclaration(targetNode.leanName);
  }

  // Resolve predecessor dependencies with full formal Lean definitions/code
  const predecessors = [];
  for (const depId of targetNode.dependencies) {
    const depNode = blueprint.nodes.find((n) => n.id === depId);
    const depTex = texEnvs.get(depId);
    let depLeanLoc: LeanDeclLocation | null = null;
    if (depNode?.leanName) {
      depLeanLoc = findLeanDeclaration(depNode.leanName);
    }

    predecessors.push({
      id: depId,
      title: depNode?.title || depTex?.title || depId,
      kind: depNode?.kind || depTex?.kind || 'declaration',
      is_verified: depNode?.isVerified ?? depTex?.isLeanok ?? false,
      lean_name: depNode?.leanName || depTex?.leanName || null,
      lean_file: depLeanLoc ? depLeanLoc.relativeFilePath : null,
      lean_code: depLeanLoc ? depLeanLoc.fullCode : null,
      latex_statement: depTex?.statement || null,
    });
  }

  return {
    id: targetNode.id,
    code_id: bountyMeta?.codeId || null,
    title: targetNode.title,
    kind: targetNode.kind,
    bounty_usd: bountyUsd,
    sponsor: bountyMeta?.sponsor || null,
    escrow_address: bountyMeta?.escrowAddress || null,
    is_verified: targetNode.isVerified,
    status: targetLeanLoc?.hasSorry ? 'has_sorry' : (targetNode.isVerified ? 'kernel_verified' : 'unproven'),
    latex_statement: texEnv?.statement || null,
    informal_proof_sketch: texEnv?.proofSketch || null,
    lean_target: {
      name: targetNode.leanName || null,
      file: targetLeanLoc ? targetLeanLoc.relativeFilePath : null,
      line_number: targetLeanLoc ? targetLeanLoc.lineNumber : null,
      source_code: targetLeanLoc ? targetLeanLoc.fullCode : null,
      has_sorry: targetLeanLoc ? targetLeanLoc.hasSorry : false,
    },
    predecessors,
    axiom_audit_policy: {
      standard_axioms: ['Classical.choice', 'Quot.sound', 'propext'],
      enforcement: 'scripts/audit_axioms.lean enforces strict zero-cheat policy.',
      note: 'Custom axioms or cheat assertions will fail the kernel audit immediately.',
    },
  };
}
