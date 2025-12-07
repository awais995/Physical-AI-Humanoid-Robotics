// build-book.js
// Script to build the complete book with validation and processing

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function validateContent() {
  console.log('Validating content for constitutional compliance...');

  // Run all validation scripts
  const validationScripts = [
    'node scripts/validate-citations.js',
    'node scripts/plagiarism-check.js',
    'node scripts/readability-check.js'
  ];

  for (const script of validationScripts) {
    try {
      console.log(`Running: ${script}`);
      const result = execSync(script, { encoding: 'utf-8' });
      console.log(result);
    } catch (error) {
      console.error(`Validation failed: ${script}`);
      console.error(error.stdout);
      console.error(error.stderr);
      throw error;
    }
  }

  console.log('All validations passed!');
  return true;
}

function buildDocusaurus() {
  console.log('Building Docusaurus site...');

  try {
    // Run docusaurus build command
    const result = execSync('npx docusaurus build', { encoding: 'utf-8' });
    console.log('Docusaurus build completed successfully');
    console.log(result);
  } catch (error) {
    console.error('Docusaurus build failed');
    console.error(error.stdout);
    console.error(error.stderr);
    throw error;
  }
}

function generatePDF() {
  console.log('Generating PDF version...');

  // Note: Actual PDF generation would require additional tools like Puppeteer
  // This is a placeholder for the PDF generation process
  console.log('PDF generation would occur here with appropriate tools');
}

function main() {
  try {
    console.log('Starting book build process...');

    // Validate all content first
    validateContent();

    // Build the Docusaurus site
    buildDocusaurus();

    // Generate PDF if needed
    generatePDF();

    console.log('Book build process completed successfully!');
  } catch (error) {
    console.error('Book build process failed:', error.message);
    process.exit(1);
  }
}

// Run the build process
main();