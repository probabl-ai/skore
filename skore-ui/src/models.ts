/**
 * A project item is a single item in the project.
 *
 * It's contributed by a user python side.
 */
export interface ProjectItem {
  media_type: string;
  value: any;
  updated_at: string;
  created_at: string;
}

/**
 * A layout is a list of keys that are visible in a view
 */
export type Layout = string[];

/**
 * A project is a collection of items and views
 */
export interface Project {
  items: { [key: string]: ProjectItem[] };
  views: { [key: string]: Layout };
}
