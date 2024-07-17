import { getAllManderPaths } from "@/services/api";
import { describe, expect, it } from "vitest";

import { createFetchResponse, mockedFetch } from "../test.utils";

describe("api", () => {
  it("Can fetch the list of manders from the server.", async () => {
    const paths = [
      "probabl-ai/demo-usecase/training/0",
      "probabl-ai/test-mandr/0",
      "probabl-ai/test-mandr/1",
      "probabl-ai/test-mandr/2",
      "probabl-ai/test-mandr/3",
      "probabl-ai/test-mandr/4",
    ];

    mockedFetch.mockResolvedValue(createFetchResponse(paths));

    const r = await getAllManderPaths();
    expect(r).toStrictEqual(paths);
  });
});
