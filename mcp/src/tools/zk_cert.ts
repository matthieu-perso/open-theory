import path from 'path';
import { execFileSync } from 'child_process';
import { SCRIPTS_DIR, getPythonBin } from '../config.js';

export const ZK_CERT_PY_PATH = path.join(SCRIPTS_DIR, 'zk_cert_tool.py');

export interface GenerateZKCertArgs {
  domain?:
    | 'OpenTheory'
    | 'OpenEconomics'
    | 'OpenTheoryAgentPassport'
    | 'OpenEconomicsAgentPassport';
  private_witness?: Record<string, number>;
  public_envelope?: Record<string, number>;
}

export interface VerifyZKCertArgs {
  certificate_json: string | Record<string, any>;
}

export function handleGenerateZKCertificate(args: GenerateZKCertArgs = {}): Record<string, any> {
  const pythonBin = getPythonBin();
  const domain = args.domain || 'OpenTheory';
  const cliArgs = [ZK_CERT_PY_PATH, 'prove', '--domain', domain];

  if (args.private_witness) {
    cliArgs.push('--private-json', JSON.stringify(args.private_witness));
  }
  if (args.public_envelope) {
    cliArgs.push('--public-json', JSON.stringify(args.public_envelope));
  }

  const stdout = execFileSync(pythonBin, cliArgs, {
    encoding: 'utf-8',
    maxBuffer: 10 * 1024 * 1024,
  });
  return JSON.parse(stdout);
}

export function handleVerifyZKCertificate(args: VerifyZKCertArgs): Record<string, any> {
  const pythonBin = getPythonBin();
  const certStr =
    typeof args.certificate_json === 'string'
      ? args.certificate_json
      : JSON.stringify(args.certificate_json);

  try {
    const stdout = execFileSync(pythonBin, [ZK_CERT_PY_PATH, 'verify', certStr], {
      encoding: 'utf-8',
      maxBuffer: 10 * 1024 * 1024,
    });
    return JSON.parse(stdout);
  } catch (err: any) {
    if (err?.stdout) {
      try {
        return JSON.parse(String(err.stdout));
      } catch {
        // fall through
      }
    }
    throw err;
  }
}
