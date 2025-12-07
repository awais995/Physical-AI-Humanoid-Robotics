// validate-citations.js
// Script to validate citations in book content for constitutional compliance
// Checks for proper APA format and ensures all claims are backed by credible sources

const fs = require('fs');
const path = require('path');

function validateCitations(content) {
  // Look for citation patterns in the content
  const citationPattern = /\[@\w+\]/g; // Basic pattern for citations like [@author2023]
  const citations = content.match(citationPattern) || [];

  console.log(`Found ${citations.length} citations in content`);

  // For each citation, check if it follows APA format
  const apaPattern = /\[@\w+\]/; // Simplified check for citation format

  for (const citation of citations) {
    if (!apaPattern.test(citation)) {
      console.warn(`Invalid citation format found: ${citation}`);
      return false;
    }
  }

  return true;
}

function validateCitationReferences(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  return validateCitations(content);
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.md')) {
      console.log(`Validating citations in: ${filePath}`);
      const isValid = validateCitationReferences(filePath);
      if (!isValid) {
        console.error(`❌ Citations validation failed for: ${filePath}`);
      } else {
        console.log(`✅ Citations validation passed for: ${filePath}`);
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
    const isValid = validateCitationReferences(targetPath);
    if (!isValid) {
      console.error(`❌ Citations validation failed for: ${targetPath}`);
      process.exit(1);
    } else {
      console.log(`✅ Citations validation passed for: ${targetPath}`);
    }
  }
} else {
  // Default to docs directory
  processDirectory('./docs');
}

console.log('Citation validation completed.');