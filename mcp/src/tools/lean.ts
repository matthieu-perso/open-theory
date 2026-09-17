import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import {
  LEAN_DIR,
  SCRIPTS_DIR,
  AUDIT_AXIOMS_PATH,
  getLakeBin,
  getLeanBin,
} from '../config.js';
import { findLeanDeclaration, LeanDeclLocation } from './blueprint.js';

export interface VerifyLeanProofArgs {
  lemma_id?: string;
  proof_code?: string;
  file_path?: string;
  run_axiom_audit?: boolean;
  dry_run?: boolean;
}

export interface CompilerMessage {
  file?: string;
  line?: number;
  column?: number;
  severity: 'error' | 'warning' | 'info';
  message: string;
}

export interface ProcessRunResult {
  exitCode: number;
  stdout: string;
  stderr: string;
  durationMs: number;
}

export interface VerificationResult {
  success: boolean;
  lake_installed: boolean;
  has_sorry: boolean;
  unauthorized_axioms: string[];
  unsolved_goals: string[];
  messages: CompilerMessage[];
  lake_build?: {
    exit_code: number;
    duration_ms: number;
    output: string;
  };
  axiom_audit?: {
    exit_code: number;
    duration_ms: number;
    output: string;
    passed: boolean;
  };
  summary: string;
  applied_to_file?: string;
}

/**
 * Executes a process with timeout and output capture.
 */
export function runCommand(
  cmd: string,
  args: string[],
  cwd: string,
  timeoutMs: number = 180000
): Promise<ProcessRunResult> {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let stdout = '';
    let stderr = '';

    const home = process.env.HOME || '';
    const elanBin = path.join(home, '.elan', 'bin');
    const customPath = `${elanBin}:${process.env.PATH || ''}`;

    const child = spawn(cmd, args, {
      cwd,
      env: {
        ...process.env,
        PATH: customPath,
      },
    });

    const timer = setTimeout(() => {
      child.kill('SIGTERM');
      stderr += `\n[Command timed out after ${timeoutMs}ms]`;
    }, timeoutMs);

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({
        exitCode: 1,
        stdout,
        stderr: `${stderr}\n${err.message}`,
        durationMs: Date.now() - startTime,
      });
    });

    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({
        exitCode: code ?? 1,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        durationMs: Date.now() - startTime,
      });
    });
  });
}

/**
 * Intelligently replaces the proof of a lemma in Lean source code.
 */
export function injectProofIntoSource(
  fileContent: string,
  shortName: string,
  proofCode: string
): { updatedContent: string; replaced: boolean } {
  const trimmedProof = proofCode.trim();

  // 1. Locate start of declaration
  const headerRegex = new RegExp(
    `(?:^|\\n)(\\s*(?:(?:protected|noncomputable|scoped|private)\\s+)*(?:@[^\\n]+\\s+)*(theorem|lemma|def)\\s+(?:[A-Za-z0-9_.]*\\.)?${shortName}\\b)`
  );
  const headerMatch = headerRegex.exec(fileContent);
  if (!headerMatch) {
    return { updatedContent: fileContent, replaced: false };
  }

  const declStart = headerMatch.index + (headerMatch[0].startsWith('\n') ? 1 : 0);

  // 2. Locate end of declaration (start of next declaration or end of OpenTheory module / file)
  const afterDeclStart = fileContent.slice(declStart);
  // Match next top-level keyword after the initial header
  const headerLength = headerMatch[1].length;
  const nextDeclRegex = /\n(?=(?:(?:protected|noncomputable|scoped|private)\s+)?(?:theorem|lemma|def|structure|inductive|class|section|end\s+OpenTheory|end\b)\b)/;
  const nextMatch = nextDeclRegex.exec(afterDeclStart.slice(headerLength));

  const declEnd = nextMatch
    ? declStart + headerLength + nextMatch.index
    : fileContent.length;

  const originalDecl = fileContent.slice(declStart, declEnd);

  // Case 1: Full theorem replacement (starts with theorem/lemma/def)
  if (/^\s*(theorem|lemma|def)\s+/m.test(trimmedProof)) {
    const updatedContent = `${fileContent.slice(0, declStart)}${trimmedProof}\n${fileContent.slice(declEnd)}`;
    return { updatedContent, replaced: true };
  }

  // Case 2: User provided ":= by ..." or ":= ..."
  if (trimmedProof.startsWith(':=')) {
    const assignIdx = originalDecl.indexOf(':=');
    if (assignIdx !== -1) {
      const declPrefix = originalDecl.slice(0, assignIdx);
      const updatedDecl = `${declPrefix}${trimmedProof}\n`;
      const updatedContent = `${fileContent.slice(0, declStart)}${updatedDecl}${fileContent.slice(declEnd)}`;
      return { updatedContent, replaced: true };
    }
  }

  // Case 3: User provided "by ..."
  if (trimmedProof.startsWith('by\n') || trimmedProof.startsWith('by ')) {
    const assignIdx = originalDecl.indexOf(':=');
    if (assignIdx !== -1) {
      const declPrefix = originalDecl.slice(0, assignIdx);
      const updatedDecl = `${declPrefix}:= ${trimmedProof}\n`;
      const updatedContent = `${fileContent.slice(0, declStart)}${updatedDecl}${fileContent.slice(declEnd)}`;
      return { updatedContent, replaced: true };
    }
  }

  // Case 4: User provided tactics or term replacing `sorry`
  // Check if declaration contains `sorry`
  const sorryRegex = /(?:[ \t]*--[^\n]*\n)*[ \t]*\bsorry\b/;
  const sorryMatch = sorryRegex.exec(originalDecl);
  if (sorryMatch) {
    const sorryStart = sorryMatch.index;
    const sorryEnd = sorryStart + sorryMatch[0].length;

    // Determine target indentation
    const linesBefore = originalDecl.slice(0, sorryStart).split('\n');
    const lastLine = linesBefore[linesBefore.length - 1];
    const indentMatch = lastLine.match(/^(\s*)/);
    const baseIndent = indentMatch ? indentMatch[1] : '  ';
    const tacticIndent = baseIndent.length > 0 ? baseIndent : '  ';

    const indentedProof = trimmedProof
      .split('\n')
      .map((line) => (line.startsWith(' ') || line.trim() === '' ? line : `${tacticIndent}${line}`))
      .join('\n');

    const updatedDecl = `${originalDecl.slice(0, sorryStart)}${indentedProof}${originalDecl.slice(sorryEnd)}`;
    const updatedContent = `${fileContent.slice(0, declStart)}${updatedDecl}${fileContent.slice(declEnd)}`;
    return { updatedContent, replaced: true };
  }

  // Case 5: Declaration has `:= by` but no explicit sorry - replace proof body after `by`
  const byIdx = originalDecl.search(/:=\s*by\b/);
  if (byIdx !== -1) {
    const byMatch = originalDecl.match(/:=\s*by\b/);
    const proofStart = byIdx + (byMatch ? byMatch[0].length : 5);
    const declPrefix = originalDecl.slice(0, proofStart);
    const indentedProof = trimmedProof
      .split('\n')
      .map((line) => (line.startsWith(' ') || line.trim() === '' ? line : `  ${line}`))
      .join('\n');
    const updatedDecl = `${declPrefix}\n${indentedProof}\n`;
    const updatedContent = `${fileContent.slice(0, declStart)}${updatedDecl}${fileContent.slice(declEnd)}`;
    return { updatedContent, replaced: true };
  }

  return { updatedContent: fileContent, replaced: false };
}

/**
 * Parses Lean 4 compiler diagnostics from lake / lean stdout and stderr.
 */
export function parseCompilerMessages(combinedOutput: string): {
  messages: CompilerMessage[];
  hasSorry: boolean;
  unsolvedGoals: string[];
} {
  const messages: CompilerMessage[] = [];
  const unsolvedGoals: string[] = [];
  let hasSorry = false;

  const lines = combinedOutput.split('\n');
  const msgRegex = /^(.+?):(\d+):(\d+):\s*(error|warning|info(?:rmation)?):\s*(.*)$/;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const match = msgRegex.exec(line);

    if (match) {
      const file = match[1];
      const lineNum = parseInt(match[2], 10);
      const colNum = parseInt(match[3], 10);
      const sevStr = match[4].toLowerCase();
      let severity: 'error' | 'warning' | 'info' = 'info';
      if (sevStr.startsWith('err')) severity = 'error';
      else if (sevStr.startsWith('warn')) severity = 'warning';

      let msgText = match[5];
      // Gather multi-line diagnostic body
      let j = i + 1;
      while (j < lines.length && !msgRegex.test(lines[j]) && !lines[j].startsWith('info:')) {
        msgText += `\n${lines[j]}`;
        j++;
      }
      i = j - 1;

      if (/unsolved goals/i.test(msgText)) {
        unsolvedGoals.push(msgText);
      }
      if (/declaration uses 'sorry'|uses sorry/i.test(msgText)) {
        hasSorry = true;
      }

      messages.push({
        file,
        line: lineNum,
        column: colNum,
        severity,
        message: msgText.trim(),
      });
    } else {
      if (/unsolved goals/i.test(line)) {
        unsolvedGoals.push(line);
      }
      if (/declaration uses 'sorry'|uses sorry/i.test(line)) {
        hasSorry = true;
      }
    }
  }

  return { messages, hasSorry, unsolvedGoals };
}

/**
 * Implementation of `verify_lean_proof` tool.
 */
export async function handleVerifyLeanProof(
  args: VerifyLeanProofArgs = {}
): Promise<VerificationResult> {
  const lakeBin = getLakeBin();
  const leanBin = getLeanBin();

  // Check if Lake or Lean is installed
  if (!lakeBin) {
    // Perform static analysis as fallback
    const staticSorry = args.proof_code ? /\bsorry\b/.test(args.proof_code) : false;
    const staticAxiom = args.proof_code ? /\baxiom\b/.test(args.proof_code) : false;
    const unauthorized = staticAxiom ? ['Static check: custom axiom keyword detected in proof'] : [];

    return {
      success: false,
      lake_installed: false,
      has_sorry: staticSorry,
      unauthorized_axioms: unauthorized,
      unsolved_goals: [],
      messages: [
        {
          severity: 'error',
          message:
            'Lake (Lean 4 build system) is not installed or not found in PATH (~/.elan/bin/lake). ' +
            'Install the toolchain with: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh',
        },
      ],
      summary:
        'Verification skipped live compilation: Lake executable not found on host. ' +
        (staticSorry ? 'Proof contains `sorry`.' : 'Static checks passed.'),
    };
  }

  let originalFileContent: string | null = null;
  let targetFilePath: string | null = null;
  let restoreNeeded = false;

  try {
    // If a proof code is provided, temporarily inject it into the target file
    if (args.proof_code && args.lemma_id) {
      const declLoc = findLeanDeclaration(args.lemma_id);
      if (!declLoc) {
        return {
          success: false,
          lake_installed: true,
          has_sorry: /\bsorry\b/.test(args.proof_code),
          unauthorized_axioms: [],
          unsolved_goals: [],
          messages: [
            {
              severity: 'error',
              message: `Could not find declaration for lemma '${args.lemma_id}' in OpenTheory library.`,
            },
          ],
          summary: `Failed to locate declaration for '${args.lemma_id}'.`,
        };
      }

      targetFilePath = declLoc.filePath;
      originalFileContent = fs.readFileSync(targetFilePath, 'utf-8');

      const { updatedContent, replaced } = injectProofIntoSource(
        originalFileContent,
        declLoc.declName,
        args.proof_code
      );

      if (!replaced) {
        return {
          success: false,
          lake_installed: true,
          has_sorry: /\bsorry\b/.test(args.proof_code),
          unauthorized_axioms: [],
          unsolved_goals: [],
          messages: [
            {
              severity: 'error',
              message: `Failed to inject proof into declaration '${declLoc.declName}'. Ensure syntax matches the theorem structure.`,
            },
          ],
          summary: `Could not replace proof in ${declLoc.relativeFilePath}.`,
        };
      }

      // Write updated content temporarily
      fs.writeFileSync(targetFilePath, updatedContent, 'utf-8');
      restoreNeeded = args.dry_run !== false; // Dry run by default: restore after test
    }

    // Step 1: Run Lake Build
    const buildRes = await runCommand(lakeBin, ['build'], LEAN_DIR);
    const combinedOutput = `${buildRes.stdout}\n${buildRes.stderr}`;
    const { messages, hasSorry: parsedSorry, unsolvedGoals } = parseCompilerMessages(combinedOutput);

    const hasProofSorry = args.proof_code ? /\bsorry\b/.test(args.proof_code) : false;
    const hasSorry = parsedSorry || hasProofSorry;
    const hasErrors = buildRes.exitCode !== 0 || messages.some((m) => m.severity === 'error') || unsolvedGoals.length > 0;

    // Step 2: Run Axiom Audit if Lake Build succeeded
    let axiomRes: ProcessRunResult | null = null;
    let unauthorizedAxioms: string[] = [];
    let auditPassed = false;

    const runAudit = args.run_axiom_audit !== false;
    if (runAudit && !hasErrors && fs.existsSync(AUDIT_AXIOMS_PATH)) {
      axiomRes = await runCommand(lakeBin, ['env', 'lean', '--run', AUDIT_AXIOMS_PATH], LEAN_DIR);
      const auditCombined = `${axiomRes.stdout}\n${axiomRes.stderr}`;

      if (axiomRes.exitCode === 0 && /\[VERIFIED\]/i.test(auditCombined)) {
        auditPassed = true;
      } else {
        auditPassed = false;
        // Parse unauthorized axioms
        const violationRegex = /Declaration:\s*([^\s]+)\s*->\s*Illegal Axiom:\s*([^\s\n]+)/g;
        let vMatch: RegExpExecArray | null;
        while ((vMatch = violationRegex.exec(auditCombined)) !== null) {
          unauthorizedAxioms.push(`${vMatch[1]}: ${vMatch[2]}`);
        }
        if (unauthorizedAxioms.length === 0) {
          unauthorizedAxioms.push('Axiom audit script reported failure');
        }
      }
    }

    const overallSuccess = !hasErrors && !hasSorry && (runAudit ? auditPassed : true);

    let summary = '';
    if (overallSuccess) {
      summary = 'Proof verified successfully by Lean 4 kernel with 0 sorry and 0 unauthorized axioms.';
    } else if (hasErrors) {
      summary = `Lean compilation failed with ${messages.filter((m) => m.severity === 'error').length} error(s).`;
    } else if (hasSorry) {
      summary = 'Proof contains open sorry goals and is not yet complete.';
    } else if (!auditPassed) {
      summary = `Axiom audit failed: ${unauthorizedAxioms.join(', ')}`;
    }

    return {
      success: overallSuccess,
      lake_installed: true,
      has_sorry: hasSorry,
      unauthorized_axioms: unauthorizedAxioms,
      unsolved_goals: unsolvedGoals,
      messages,
      lake_build: {
        exit_code: buildRes.exitCode,
        duration_ms: buildRes.durationMs,
        output: combinedOutput.trim(),
      },
      axiom_audit: axiomRes
        ? {
            exit_code: axiomRes.exitCode,
            duration_ms: axiomRes.durationMs,
            output: `${axiomRes.stdout}\n${axiomRes.stderr}`.trim(),
            passed: auditPassed,
          }
        : undefined,
      summary,
      applied_to_file: targetFilePath ? path.relative(LEAN_DIR, targetFilePath) : undefined,
    };
  } finally {
    // Safely restore original file content if dry run
    if (restoreNeeded && targetFilePath && originalFileContent !== null) {
      fs.writeFileSync(targetFilePath, originalFileContent, 'utf-8');
    }
  }
}
