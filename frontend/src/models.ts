export type KeyLayoutSize = "small" | "medium" | "large";
export type KeyMoveDirection = "up" | "down";

export type Layout = Array<{ key: string; size: KeyLayoutSize }>;

export interface ReportItem {
  media_type: string;
  value: any;
  updated_at: string;
  created_at: string;
}

export interface Report {
  items: { [key: string]: ReportItem };
  layout: Layout;
}
