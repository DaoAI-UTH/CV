# Report Outline

Use this as the writing structure for the final report.

## 1. Problem Statement

- Problem: detect and recognize container identification codes from real container images.
- Hypothesis: contrast enhancement, adaptive thresholding, Canny edges, and contour
  filtering will localize code regions; segmented characters can be recognized with KNN.
- Success criteria:
  - detection precision and recall at IoU 0.5
  - mean IoU
  - character accuracy on a manually labeled subset

## 2. Related Work

Include at least three papers/systems about OCR, container code recognition, scene text
detection, or classical OCR pipelines.

## 3. Method

Pipeline:

1. Convert image to HSV value channel or grayscale.
2. Enhance local contrast with Top Hat and Black Hat morphology.
3. Reduce noise with Gaussian filtering.
4. Binarize with adaptive Gaussian thresholding.
5. Detect edges with Canny.
6. Find and filter contours by area and aspect ratio.
7. Crop candidate code regions.
8. Segment characters by contours.
9. Recognize characters with OpenCV KNN.

Explain why each step is suitable for container codes.

## 4. Experiments and Results

Show intermediate images:

- original image
- grayscale/HSV value channel
- contrast image
- blurred image
- binary image
- edge image
- final detected region
- segmented characters

Show parameter sweeps:

- Gaussian kernel: 3, 5, 9
- adaptive threshold C: 3, 9, 15
- Canny low threshold: 30, 60, 100
- morphology kernel: 9, 17, 31

Report quantitative metrics:

- precision
- recall
- F1
- mean IoU
- character accuracy if a labeled subset is available

## 5. Discussion

- Which images worked well?
- Which cases failed: low contrast, occlusion, angled text, vertical text, rusty container
  surfaces, background edges.
- Compare results with the hypothesis.

## 6. Conclusion

Summarize what the pipeline achieved and what should be improved.

## 7. Appendix

Put large result figures and sweep panels here.

## 8. References

Use IEEE or APA format. Minimum five sources.
