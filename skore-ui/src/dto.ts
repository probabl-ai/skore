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

export type Favorability = "greater_is_better" | "lower_is_better" | "unknown";

export interface ScalarResultDto {
  name: string;
  value: number;
  stddev?: number;
  label?: string;
  description?: string;
  favorability: Favorability;
}

export interface TabularResultDto {
  name: string;
  columns: any[];
  data: any[][];
  favorability: Favorability[];
}

export interface PrimaryResultsDto {
  scalarResults: ScalarResultDto[];
  tabularResults: TabularResultDto[];
}

export interface PlotDto {
  name: string;
  value: any;
}

export interface DetailSectionItemDto {
  name: string;
  description: string;
  value: string;
}

export interface DetailSectionDto {
  title: string;
  icon: string;
  items: DetailSectionItemDto[];
}
