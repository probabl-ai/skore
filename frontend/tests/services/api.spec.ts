import { fetchAllManderUris, fetchMander, fetchShareableBlob, putLayout } from "@/services/api";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DataStore, type Layout } from "@/models";
import { createFetchResponse, mockedFetch } from "../test.utils";

describe("API Service", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can fetch the list of manders from the server.", async () => {
    const uris = [
      "probabl-ai/demo-usecase/training/0",
      "probabl-ai/test-skore/0",
      "probabl-ai/test-skore/1",
      "probabl-ai/test-skore/2",
      "probabl-ai/test-skore/3",
      "probabl-ai/test-skore/4",
    ];

    mockedFetch.mockResolvedValue(createFetchResponse(uris));

    const r = await fetchAllManderUris();
    expect(r).toStrictEqual(uris);
  });

  it("Can fetch a mander from the server", async () => {
    const mander = {
      schema: "schema:dashboard:v0",
      uri: "probal-ai/demo-usecase/training/1",
      payload: {
        title: {
          type: "string",
          data: "My Awesome Dashboard",
        },
        errors: {
          type: "array",
          data: [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        "creation date": {
          type: "date",
          data: "2024-07-24",
        },
        "last updated": {
          type: "datetime",
          data: "2024-07-24T11:31:00Z",
        },
        score: {
          type: "number",
          data: 0.87,
        },
        count: {
          type: "integer",
          data: 234567,
        },
        monitoring: {
          type: "markdown",
          data: "- The fitting run used **92.24347826086958%** of your CPU (min: 0.0%; max: 100.4%)\n- The fitting run used **0.7128300874129586%** of your RAM (min: 0.7058143615722656%; max: 0.7147789001464844%)",
        },
        "custom html": {
          type: "html",
          data: "<div class=container><div id=square></div></div><script>const square = document.getElementById('square');setInterval(() => {square.style.backgroundColor = '#' + (Math.random() * 0xFFFFFF << 0).toString(16);}, 500)</script><style>.container{display:flex;justify-content:center;align-items:center;width:100px;height:100px;padding:5px;background-color:#fff;border:dashed 1px #ccc;height:50px;transition:background-color linear .5s}</style>",
        },
      },
    };
    mockedFetch.mockResolvedValue(createFetchResponse(mander));

    const r = await fetchMander("random");
    expect(r).toBeInstanceOf(DataStore);
    expect(r?.infoKeys.length).toBe(8);
  });

  it("Can persist a layout.", async () => {
    const layout: Layout = [
      { key: "title", size: "medium" },
      { key: "errors", size: "large" },
      { key: "creation_date", size: "small" },
    ];
    const mander = new DataStore(
      "random",
      {
        title: {
          type: "string",
          data: "My Awesome Dashboard",
        },
        errors: {
          type: "array",
          data: [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        creation_date: {
          type: "date",
          data: "2024-07-24",
        },
      },
      layout
    );

    mockedFetch.mockResolvedValue(createFetchResponse(mander, 201));

    const r = await putLayout("random", layout);
    expect(r).toBeInstanceOf(DataStore);
    expect(r).toEqual(mander);
    expect(r?.layout).toEqual(layout);
  });

  it("Can report errors.", async () => {
    const error = new Error("Something went wrong");
    mockedFetch.mockImplementation(() => {
      throw error;
    });

    expect(await fetchAllManderUris()).toEqual([]);
    expect(await fetchMander("random")).toBeNull();
    expect(await putLayout("random", [])).toBeNull();
    expect(await fetchShareableBlob("random")).toBeNull();
  });

  it("Can fetch a shareable blob.", async () => {
    const blob = new Blob(["Hello, world!"], { type: "text/plain" });
    mockedFetch.mockResolvedValue(createFetchResponse(blob));

    const r = await fetchShareableBlob("random");
    expect(r).toBeInstanceOf(Blob);
    expect(r?.size).toBe(13);
  });
});
