// validate-constitutional-compliance.js
// Script to validate content for constitutional compliance
// Checks for adherence to project principles and standards

const fs = require('fs');
const path = require('path');

// Constitutional compliance checks
const constitutionalChecks = {
  // Check for proper academic tone and structure
  academicTone: (content) => {
    const issues = [];

    // Check for informal language that doesn't belong in academic content
    const informalPatterns = [
      /\b(guys|dude|cool|awesome|OMG|LOL|etc\.)\b/gi,
      /(^|\n)\s*[>#]\s*(joke|fun|just kidding)/gi
    ];

    for (const pattern of informalPatterns) {
      const matches = content.match(pattern);
      if (matches) {
        issues.push(`Found informal language: ${matches.slice(0, 3).join(', ')}`);
      }
    }

    return issues;
  },

  // Check for proper technical accuracy indicators
  technicalAccuracy: (content) => {
    const issues = [];

    // Check for claims without evidence
    const unsupportedClaimPattern = /\b(apparently|maybe|possibly|might be|could be)\s+(the case|true|correct)\b/gi;
    const unsupportedMatches = content.match(unsupportedClaimPattern);
    if (unsupportedMatches) {
      issues.push(`Found potentially unsupported claims: ${unsupportedMatches.slice(0, 3).join(', ')}`);
    }

    return issues;
  },

  // Check for proper citation format and usage
  citationFormat: (content) => {
    const issues = [];

    // Check for inline claims without citations
    const claimWithoutCitationPattern = /\b(according to|research shows|studies indicate|experts say)\b/gi;
    const claims = content.match(claimWithoutCitationPattern);
    if (claims) {
      issues.push(`Found claims that may need citations: ${claims.slice(0, 3).join(', ')}`);
    }

    // Check for citation format
    const citationPattern = /\[@\w+\]/g; // Basic pattern for citations like [@author2023]
    const citations = content.match(citationPattern) || [];
    console.log(`Found ${citations.length} citations in content`);

    return issues;
  },

  // Check for proper structure and organization
  contentStructure: (content) => {
    const issues = [];

    // Check for proper headings hierarchy
    const headingPattern = /^#+\s+/gm;
    const headings = content.match(headingPattern) || [];

    // Basic check: ensure there are appropriate section headers
    if (headings.length < 2) {
      issues.push('Content may need better structural organization with appropriate headings');
    }

    return issues;
  },

  // Check for ethical considerations in AI/Robotics content
  ethicalConsiderations: (content) => {
    const issues = [];

    // Check for mention of ethical implications in AI/Robotics content
    const aiEthicsKeywords = ['ethics', 'bias', 'privacy', 'safety', 'responsibility', 'fairness'];
    const hasEthicsContent = aiEthicsKeywords.some(keyword =>
      content.toLowerCase().includes(keyword)
    );

    if (!hasEthicsContent) {
      issues.push('AI/Robotics content should include ethical considerations');
    }

    return issues;
  }
};

function validateConstitutionalCompliance(content, filePath) {
  console.log(`Validating constitutional compliance for: ${filePath}`);

  const allIssues = [];

  // Run each constitutional check
  for (const [checkName, checkFunction] of Object.entries(constitutionalChecks)) {
    try {
      const issues = checkFunction(content);
      if (issues.length > 0) {
        allIssues.push(...issues);
        console.log(`  ${checkName}: ${issues.length} issues found`);
      }
    } catch (error) {
      console.error(`Error in ${checkName} validation:`, error.message);
      allIssues.push(`Validation error in ${checkName}: ${error.message}`);
    }
  }

  if (allIssues.length === 0) {
    console.log(`✅ Constitutional compliance validation passed for: ${filePath}`);
    return true;
  } else {
    console.log(`❌ Constitutional compliance validation found ${allIssues.length} issues in: ${filePath}`);
    for (const issue of allIssues) {
      console.log(`  - ${issue}`);
    }
    return false;
  }
}

function validateFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  return validateConstitutionalCompliance(content, filePath);
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);
  let allValid = true;

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      const dirValid = processDirectory(filePath);
      allValid = allValid && dirValid;
    } else if (file.endsWith('.md')) {
      const fileValid = validateFile(filePath);
      allValid = allValid && fileValid;
    }
  }

  return allValid;
}

// Main execution
if (process.argv[2]) {
  const targetPath = process.argv[2];
  const stat = fs.statSync(targetPath);

  if (stat.isDirectory()) {
    const isValid = processDirectory(targetPath);
    if (!isValid) {
      console.error('❌ Constitutional compliance validation failed for directory');
      process.exit(1);
    } else {
      console.log('✅ Constitutional compliance validation passed for directory');
    }
  } else if (stat.isFile()) {
    const isValid = validateFile(targetPath);
    if (!isValid) {
      console.error('❌ Constitutional compliance validation failed for file');
      process.exit(1);
    }
  }
} else {
  // Default to docs directory
  const isValid = processDirectory('./docs');
  if (!isValid) {
    console.error('❌ Constitutional compliance validation failed for docs directory');
    process.exit(1);
  }
}

console.log('Constitutional compliance validation completed.');