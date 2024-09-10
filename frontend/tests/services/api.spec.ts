import { fetchReport } from "@/services/api";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createFetchResponse, mockedFetch } from "../test.utils";

describe("API Service", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can fetch the current project from the server", async () => {
    const p = {
      Any: { item_type: "json", media_type: null, serialized: { k1: "v1" } },
      Array: {
        item_type: "json",
        media_type: null,
        serialized: [1, 2, 3],
      },
    };
    mockedFetch.mockResolvedValue(createFetchResponse(p));

    const r = await fetchReport();
    expect(Object.keys(r!).length).toBe(2);
  });

  it("Can report errors.", async () => {
    const error = new Error("Something went wrong");
    mockedFetch.mockImplementation(() => {
      throw error;
    });

    expect(await fetchReport()).toBeNull();
  });
});
