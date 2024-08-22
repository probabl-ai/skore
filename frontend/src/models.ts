export type ItemType =
  | "boolean"
  | "integer"
  | "number"
  | "string"
  | "any"
  | "array"
  | "date"
  | "datetime"
  | "file"
  | "html"
  | "markdown"
  | "vega"
  | "dataframe"
  | "image"
  | "cv_results"
  | "numpy_array";

export interface IPayloadItemMetadata {
  display_type: string;
  created_at: string;
  updated_at: string;
}

export interface IPayloadItem {
  type: ItemType;
  data: any;
  metadata?: IPayloadItemMetadata;
}

export type KeyLayoutSize = "small" | "medium" | "large";

export type Layout = Array<{ key: string; size: KeyLayoutSize }>;

export class DataStore {
  constructor(
    public uri: string,
    public payload: { [key: string]: IPayloadItem },
    public layout: Layout
  ) {}

  get plotKeys(): string[] {
    return this._getKeysByType(["vega"]);
  }

  get artifactKeys(): string[] {
    return this._getKeysByType(["file"]);
  }

  get infoKeys(): string[] {
    return this._getKeysByType([
      "boolean",
      "integer",
      "number",
      "string",
      "any",
      "array",
      "date",
      "datetime",
      "html",
      "markdown",
      "dataframe",
      "image",
      "cv_results",
      "numpy_array",
    ]);
  }

  get(key: string): IPayloadItem {
    return this.payload[key];
  }

  _getKeysByType(types: ItemType[]) {
    const filteredKeys = [];
    for (const [key, item] of Object.entries(this.payload)) {
      if (types.includes(item.type)) {
        filteredKeys.push(key);
      }
    }
    return filteredKeys;
  }
}
