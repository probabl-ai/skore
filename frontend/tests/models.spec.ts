import { type ItemType } from "@/models";
import { describe, expect, it } from "vitest";
import { makeDataStore } from "./test.utils";

describe("DataStore model", () => {
  it("Can access keys by type", () => {
    const infoKeys: ItemType[] = [
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
      "sklearn_model",
    ];
    const plotKeys: ItemType[] = ["vega", "matplotlib_figure"];

    const m = makeDataStore("/test/fixture", [...infoKeys, ...plotKeys]);

    expect(m.infoKeys).toHaveLength(infoKeys.length);
    expect(m.plotKeys).toHaveLength(plotKeys.length);
    expect(m.artifactKeys).toHaveLength(0);

    expect(m.get("boolean")).toBeDefined();
  });
});
