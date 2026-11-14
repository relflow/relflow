/** Presentation nodes describe a hierarchy without interpreting values or schemas. */
export interface TreeNode {
  label: string;
  detail?: string;
  children?: readonly TreeNode[];
}
