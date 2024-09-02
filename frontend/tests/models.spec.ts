import { DataStore, type IPayloadItem, type ItemType } from "@/models";
import { describe, expect, it } from "vitest";

function makePayloadItem(type: ItemType, data: any = {}): IPayloadItem {
  const now = new Date().toISOString();
  return {
    type,
    data,
    metadata: {
      display_type: type,
      created_at: now,
      updated_at: now,
    },
  };
}

describe("models", () => {
  it("Can access keys by type", async () => {
    const m = new DataStore(
      "/test/fixture",
      {
        boolean: makePayloadItem("boolean"),
        integer: makePayloadItem("integer"),
        number: makePayloadItem("number"),
        string: makePayloadItem("string"),
        any: makePayloadItem("any"),
        array: makePayloadItem("array"),
        date: makePayloadItem("date"),
        datetime: makePayloadItem("datetime"),
        html: makePayloadItem("html"),
        markdown: makePayloadItem("markdown"),
        dataframe: makePayloadItem("dataframe"),
        image: makePayloadItem("image"),
        cv_results: makePayloadItem("cv_results"),
        numpy_array: makePayloadItem("numpy_array"),
        sklearn_model: makePayloadItem("sklearn_model"),
        vega: makePayloadItem("vega"),
        matplotlib_figure: makePayloadItem("matplotlib_figure"),
      },
      []
    );

    expect(m.infoKeys).toHaveLength(15);
    expect(m.plotKeys).toHaveLength(2);
    expect(m.artifactKeys).toHaveLength(0);

    expect(m.get("boolean")).toBeDefined();
  });
});
