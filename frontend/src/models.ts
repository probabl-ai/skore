export type KeyLayoutSize = "small" | "medium" | "large";
export type KeyMoveDirection = "up" | "down";

export type Layout = Array<{ key: string; size: KeyLayoutSize }>;

export interface ReportItem {
  item_type: string;
  media_type: string | null;
  serialized: any;
  updated_at: string;
  created_at: string;
}

export interface Report {
  items: { [key: string]: ReportItem };
  layout: Layout;
}

export type SupportedImageMimeType = "image/svg+xml" | "image/png" | "image/jpeg" | "image/webp";
