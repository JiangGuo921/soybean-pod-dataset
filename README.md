# Soybean Pod Detection Dataset

A field-grown soybean pod detection dataset with bounding-box annotations,
collected from 113 diverse soybean accessions.

## Overview

The dataset comprises 1,130 RGB images collected from 113 soybean accessions
(10 plants per accession, one image per plant), grown in 2025 at the Chongqing
Sub-center for Plant Variety Testing, Ministry of Agriculture and Rural Affairs,
China. The accessions span diverse plant architectures (erect, semi-erect, and
semi-prostrate; plant height 20–130 cm) and pod densities (sparse to dense).

The single
annotated class is `pod`. Images were randomly split into training, validation,
and test sets at a 7:2:1 ratio.

| Split | Images | Description        |
|-------|--------|--------------------|
| train | 780    | Training set       |
| val   | 222    | Validation set     |
| test  | 111    | Test set           |

## Dataset Structure

Images and their annotation files share the same folder and file name:
