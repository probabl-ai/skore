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
  note?: string;
  version: number;
}

/**
 * An activity feed is a list of item.
 *
 * Sorted from newest to oldest.
 */
export type ActivityFeedDto = ProjectItemDto[];

/**
 * Project info
 */
export interface ProjectInfoDto {
  name: string;
  path: string;
}
