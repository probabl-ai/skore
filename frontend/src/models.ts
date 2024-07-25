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
  | "vega";

export interface IPayloadItem {
  type: ItemType;
  data: any;
}

export class DataStore {
  constructor(
    public uri: string,
    public payload: { [key: string]: IPayloadItem }
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
    ]);
  }

  get(key: string): any {
    return this.payload[key].data;
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
