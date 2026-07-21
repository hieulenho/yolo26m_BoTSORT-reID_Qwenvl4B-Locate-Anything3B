# Semantic GT review: wildlife_black_noddies

1. Watch the source video and inspect every image in `contact_sheets/`.
2. Fill `class_label`, optional `fine_label`, `review_status=reviewed`, and `annotator`
   for every row in `track_annotations.csv`. Use `ignore=true` only when no human can judge it.
3. Review `domain`, `detector_route`, and the complete object list in
   `ground_truth_review.yaml`; then set its review status and provenance.
4. Run the finalize command. Draft/model proposals cannot be evaluated as GT.
