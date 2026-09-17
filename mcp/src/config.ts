import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Discovers the OpenTheory repository root directory.
 */
function findWorkspaceRoot(): string {
  if (process.env.OPEN_THEORY_ROOT && fs.existsSync(process.env.OPEN_THEORY_ROOT)) {
    return path.resolve(process.env.OPEN_THEORY_ROOT);
  }

  // Walk up from current working directory
  let current = process.cwd();
  for (let i = 0; i < 6; i++) {
    if (
      fs.existsSync(path.join(current, 'lakefile.lean')) ||
      fs.existsSync(path.join(current, 'blueprint', 'blueprint.json')) ||
      fs.existsSync(path.join(current, 'OpenTheory'))
    ) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }

  // Walk up from __dirname
  current = __dirname;
  for (let i = 0; i < 6; i++) {
    if (
      fs.existsSync(path.join(current, 'lakefile.lean')) ||
      fs.existsSync(path.join(current, 'blueprint', 'blueprint.json')) ||
      fs.existsSync(path.join(current, 'OpenTheory'))
    ) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }

  // Fallback: two directories up from mcp/src or mcp/dist
  return path.resolve(__dirname, '..', '..');
}

export const WORKSPACE_ROOT = findWorkspaceRoot();

// Blueprint Paths
export const BLUEPRINT_DIR = path.join(WORKSPACE_ROOT, 'blueprint');
export const BLUEPRINT_JSON_PATH = path.join(BLUEPRINT_DIR, 'blueprint.json');
export const BLUEPRINT_TEX_PATH = path.join(BLUEPRINT_DIR, 'src', 'content.tex');

// Lean 4 Paths
// In some setups Lean is located at root, in others inside a lean/ subfolder
function findLeanDir(): string {
  const subLean = path.join(WORKSPACE_ROOT, 'lean');
  if (fs.existsSync(path.join(subLean, 'lakefile.lean'))) {
    return subLean;
  }
  return WORKSPACE_ROOT;
}

export const LEAN_DIR = findLeanDir();
export const LAKEFILE_PATH = path.join(LEAN_DIR, 'lakefile.lean');
export const OPENTHEORY_LEAN_DIR = path.join(LEAN_DIR, 'OpenTheory');

// Scripts Paths
function findScriptsDir(): string {
  if (fs.existsSync(path.join(LEAN_DIR, 'scripts'))) {
    return path.join(LEAN_DIR, 'scripts');
  }
  return path.join(WORKSPACE_ROOT, 'scripts');
}

export const SCRIPTS_DIR = findScriptsDir();
export const AUDIT_AXIOMS_PATH = path.join(SCRIPTS_DIR, 'audit_axioms.lean');
export const EXPORT_BLUEPRINT_PY_PATH = path.join(SCRIPTS_DIR, 'export_blueprint.py');

/**
 * Locate executable binaries with support for custom env vars, Elan paths, Homebrew, and PATH.
 */
function isExecutable(filePath: string): boolean {
  try {
    fs.accessSync(filePath, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

export function findBinary(binaryName: string, extraSearchPaths: string[] = []): string | null {
  const envVar = `${binaryName.toUpperCase()}_PATH`;
  if (process.env[envVar] && fs.existsSync(process.env[envVar]!)) {
    return process.env[envVar]!;
  }

  const home = process.env.HOME || '';
  const searchCandidates = [
    ...extraSearchPaths,
    path.join(home, '.elan', 'bin', binaryName),
    path.join('/opt/homebrew/bin', binaryName),
    path.join('/usr/local/bin', binaryName),
    path.join('/usr/bin', binaryName),
    path.join('/bin', binaryName),
  ];

  for (const candidate of searchCandidates) {
    if (candidate && fs.existsSync(candidate) && isExecutable(candidate)) {
      return candidate;
    }
  }

  // Check system PATH
  const envPath = process.env.PATH || '';
  for (const dir of envPath.split(path.delimiter)) {
    if (!dir) continue;
    const full = path.join(dir, binaryName);
    if (fs.existsSync(full) && isExecutable(full)) {
      return full;
    }
  }

  return null;
}

export function getLakeBin(): string | null {
  return findBinary('lake');
}

export function getLeanBin(): string | null {
  return findBinary('lean');
}

export function getPythonBin(): string {
  return findBinary('python3') || findBinary('python') || 'python3';
}

export function getGitBin(): string {
  return findBinary('git') || 'git';
}

/**
 * Canonical fallback bounty values and sponsor metadata
 * Derived from docs/overview.md, content.tex, and Lean source annotations.
 */
export interface BountyMetadata {
  amountUsd: number;
  sponsor: string;
  codeId?: string;
  status: 'funded' | 'in_review' | 'claimed';
  escrowAddress?: string;
}

export const KNOWN_BOUNTIES: Record<string, BountyMetadata> = {
  'lem:spectral_bias_mup': {
    amountUsd: 2500,
    sponsor: 'Open Philanthropy AI Safety Fund',
    codeId: 'LEM-2.1',
    status: 'funded',
    escrowAddress: '0x8849F6cD2dE4C69b4F24F21A37F5Ec64A2a4e402',
  },
  'lem:representation_contraction': {
    amountUsd: 1500,
    sponsor: 'Schmidt Sciences AI2050 Pool',
    codeId: 'LEM-2.2',
    status: 'funded',
    escrowAddress: '0x17a6A8E3516591Dbb8F0C6D1F97B67B78a6F50B9',
  },
  'lem:ntk_stationarity': {
    amountUsd: 1000,
    sponsor: 'OpenTheory Treasury',
    codeId: 'LEM-2.3',
    status: 'funded',
    escrowAddress: '0x43b2C9a5E6102Fb8923a96860E87F63d274B5327',
  },
  'lem:hessian_spectral': {
    amountUsd: 2000,
    sponsor: 'Anthropic Alignment Research',
    codeId: 'LEM-2.4',
    status: 'funded',
    escrowAddress: '0x71C5A82B149F446266D0C9298B359e1981F3eF1C',
  },
  'thm:asymptotic_ntk': {
    amountUsd: 3000,
    sponsor: 'Deep Foundations Fellowship',
    codeId: 'THM-3.1',
    status: 'funded',
    escrowAddress: '0x992B104F5C98e1a660a9f5d134C286a11e8C1F45',
  },
  'thm:feature_learning_mup': {
    amountUsd: 4500,
    sponsor: 'Simons Foundation Deep Learning Pool',
    codeId: 'THM-3.2',
    status: 'funded',
    escrowAddress: '0x66B72D3b8F1F2746C8987E236a284F4aF2E23cD1',
  },
  'conj:grand_feature_learning': {
    amountUsd: 15000,
    sponsor: 'OpenTheory Foundation Consortium',
    codeId: 'CONJ-01',
    status: 'funded',
    escrowAddress: '0x52E92543d3Ac76C488F3C59d43501E852D082987',
  },
};
