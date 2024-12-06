/**
 * A project item is a single item in the project.
 *
 * It's contributed by a user python side.
 */
export interface ProjectItemDto {
  name: string;
  media_type: string;
  value: any;
  updated_at: string;
  created_at: string;
}

/**
 * A layout is a list of keys that are visible in a view
 */
export type LayoutDto = string[];

/**
 * A project is a collection of items and views
 */
export interface ProjectDto {
  items: { [key: string]: ProjectItemDto[] };
  views: { [key: string]: LayoutDto };
}

/**
 * An activity feed is a list of item.
 *
 * Sorted from newest to oldest.
 */
export type ActivityFeedDto = ProjectItemDto[];
