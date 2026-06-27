export interface GreeneryClass {
  id: string;
  label: string;
  description: string;
  color: string;
}

export const GREENERY_CLASSES: GreeneryClass[] = [
  {
    id: "dense_green",
    label: "Dense Green Cover",
    description: "High vegetation — forests, agriculture, golf courses",
    color: "#166534",
  },
  {
    id: "moderate_green",
    label: "Moderate Green Cover",
    description: "Moderate vegetation — chaparral, riversides",
    color: "#22c55e",
  },
  {
    id: "sparse_green",
    label: "Sparse Green Cover",
    description: "Low urban greenery — residential areas with scattered trees",
    color: "#84cc16",
  },
  {
    id: "minimal_green",
    label: "Minimal Green Cover",
    description: "Built-up areas — buildings, roads, parking lots",
    color: "#78716c",
  },
];

export const IMAGE_SIZE = 224;
