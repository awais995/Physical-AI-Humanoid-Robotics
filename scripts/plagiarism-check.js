// plagiarism-check.js
// Script to check content for plagiarism and ensure 0% tolerance as required by constitution

const fs = require('fs');
const path = require('path');

function checkPlagiarism(content, filePath) {
  // Basic check for common problematic patterns that might indicate plagiarism
  // This is a simplified version - a full implementation would require external services
  const checks = {
    excessiveQuotedText: false,
    missingCitations: false,
    suspiciousPatterns: []
  };

  // Check for large blocks of quoted text without citations
  const quoteBlocks = content.match(/"{3}[\s\S]*?"{3}|`{3}[\s\S]*?`{3}/g) || [];
  for (const block of quoteBlocks) {
    if (block.length > 200) { // If quote block is larger than 200 characters
      if (!hasCitationNearby(content, content.indexOf(block))) {
        checks.suspiciousPatterns.push('Large quoted block without nearby citation');
        checks.excessiveQuotedText = true;
      }
    }
  }

  // Check for text blocks that might be copied without attribution
  const paragraphs = content.split(/\n\s*\n/);
  for (const para of paragraphs) {
    if (para.length > 300) { // Large paragraph check
      // Look for common phrases that might indicate copied content
      const commonPhrases = [
        /according to research/i,
        /studies show/i,
        /it has been proven/i,
        /experts say/i
      ];

      for (const phrase of commonPhrases) {
        if (phrase.test(para) && !hasCitationNearby(content, content.indexOf(para))) {
          checks.suspiciousPatterns.push('Unattributed claim with common academic phrase');
        }
      }
    }
  }

  const hasIssues = checks.excessiveQuotedText || checks.missingCitations || checks.suspiciousPatterns.length > 0;

  if (hasIssues) {
    console.log(`⚠️ Potential plagiarism issues found in ${filePath}:`);
    checks.suspiciousPatterns.forEach(pattern => console.log(`  - ${pattern}`));
    return false;
  }

  console.log(`✅ Plagiarism check passed for: ${filePath}`);
  return true;
}

function hasCitationNearby(content, position) {
  // Check for citations within 100 characters before or after position
  const start = Math.max(0, position - 100);
  const end = Math.min(content.length, position + 100);
  const nearbyText = content.substring(start, end);

  // Look for citation patterns
  return /(@\w+\b|\[@\w+\]|cited|reference|source)/i.test(nearbyText);
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.md')) {
      console.log(`Checking plagiarism in: ${filePath}`);
      const content = fs.readFileSync(filePath, 'utf8');
      const isClean = checkPlagiarism(content, filePath);
      if (!isClean) {
        console.error(`❌ Plagiarism check failed for: ${filePath}`);
      }
    }
  }
}

// Main execution
if (process.argv[2]) {
  const targetPath = process.argv[2];
  const stat = fs.statSync(targetPath);

  if (stat.isDirectory()) {
    processDirectory(targetPath);
  } else if (stat.isFile()) {
    const content = fs.readFileSync(targetPath, 'utf8');
    const isClean = checkPlagiarism(content, targetPath);
    if (!isClean) {
      console.error(`❌ Plagiarism check failed for: ${targetPath}`);
      process.exit(1);
    }
  }
} else {
  // Default to docs directory
  processDirectory('./docs');
}

console.log('Plagiarism check completed.');