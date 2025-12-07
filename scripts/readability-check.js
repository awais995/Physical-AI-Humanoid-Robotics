// readability-check.js
// Script to analyze content readability and ensure it meets grade 10-12 level requirement

const fs = require('fs');
const path = require('path');

function countSyllables(word) {
  word = word.toLowerCase();
  if (word.length <= 3) return 1;
  word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
  word = word.replace(/^y/, '');
  const matches = word.match(/[aeiouy]{1,2}/g);
  return matches ? matches.length : 1;
}

function calculateFleschKincaidGradeLevel(content) {
  // Remove markdown formatting for analysis
  const cleanText = content
    .replace(/[#*\-_`[\]]/g, '') // Remove basic markdown
    .replace(/\[.*?\]\(.*?\)/g, '') // Remove links
    .replace(/!\[.*?\]\(.*?\)/g, ''); // Remove images

  const sentences = cleanText.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const words = cleanText.split(/\s+/).filter(w => w.length > 0);
  const syllables = words.map(word => countSyllables(word)).reduce((a, b) => a + b, 0);

  if (sentences.length === 0 || words.length === 0) {
    console.log('⚠️ Cannot calculate readability for empty content');
    return 0;
  }

  // Flesch-Kincaid Grade Level formula:
  // 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
  const avgWordsPerSentence = words.length / sentences.length;
  const avgSyllablesPerWord = syllables / words.length;

  const gradeLevel = (0.39 * avgWordsPerSentence) + (11.8 * avgSyllablesPerWord) - 15.59;

  return Math.max(0, gradeLevel); // Ensure non-negative result
}

function checkReadability(content, filePath) {
  const gradeLevel = calculateFleschKincaidGradeLevel(content);

  console.log(`Readability grade level for ${filePath}: ${gradeLevel.toFixed(2)}`);

  // Check if grade level is within the required range (10-12)
  if (gradeLevel >= 10 && gradeLevel <= 12) {
    console.log(`✅ Readability check passed for: ${filePath} (Grade level: ${gradeLevel.toFixed(2)})`);
    return true;
  } else if (gradeLevel < 10) {
    console.log(`⚠️ Content may be too simple for target audience (Grade level: ${gradeLevel.toFixed(2)})`);
    return false;
  } else {
    console.log(`⚠️ Content may be too complex for target audience (Grade level: ${gradeLevel.toFixed(2)})`);
    return false;
  }
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.md')) {
      console.log(`Checking readability in: ${filePath}`);
      const content = fs.readFileSync(filePath, 'utf8');
      const isReadable = checkReadability(content, filePath);
      if (!isReadable) {
        console.error(`❌ Readability check failed for: ${filePath}`);
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
    const isReadable = checkReadability(content, targetPath);
    if (!isReadable) {
      console.error(`❌ Readability check failed for: ${targetPath}`);
      process.exit(1);
    }
  }
} else {
  // Default to docs directory
  processDirectory('./docs');
}

console.log('Readability check completed.');