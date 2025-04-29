# scripts/run_clean_archive.sh
#!/bin/bash

echo "🏙️ Running Chicago Population Pipeline..."
python main.py

if [ $? -eq 0 ]; then
  echo "✅ Pipeline run successful!"

  echo "🧹 Cleaning duplicate outputs..."
  python scripts/clean_duplicates.py

  echo "📦 Archiving run outputs..."
  TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
  zip -r "output/run_archives/run_archive_$TIMESTAMP.zip" output/* -x "output/run_archives/*"

  echo "📝 Saving pipeline summary..."
  echo "# Pipeline Run Summary" > output/pipeline_summary.md
  echo "**Timestamp:** $TIMESTAMP" >> output/pipeline_summary.md
  echo "**Status:** Successful" >> output/pipeline_summary.md

  echo "🎉 All done!"
else
  echo "❌ Pipeline failed. No cleaning or archiving done."
fi