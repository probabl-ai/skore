import FileTree from "@/components/FileTree.vue";
import DashboardView from "@/views/DashboardView.vue";
import { describe, expect, it } from "vitest";
import { createFetchResponse, mockedFetch, mountSuspense } from "../test.utils";

describe("DashboardView", () => {
  it("Parse the list of fetched list of mander to an array of FileTreeNode.", async () => {
    const paths = [
      "probabl-ai/demo-usecase/training/0",
      "probabl-ai/test-mandr/0",
      "probabl-ai/test-mandr/1",
      "probabl-ai/test-mandr/2",
      "probabl-ai/test-mandr/3",
      "probabl-ai/test-mandr/4",
    ];

    mockedFetch.mockResolvedValue(createFetchResponse(paths));

    const wrapper = await mountSuspense(DashboardView);
    const fileTree = await wrapper.getComponent(FileTree);
    expect(fileTree.props()).toEqual({
      nodes: [
        {
          label: "probabl-ai",
          children: [
            {
              label: "demo-usecase",
              children: [{ label: "training", children: [{ label: "0" }] }],
            },
            {
              label: "test-mandr",
              children: [
                { label: "0" },
                { label: "1" },
                { label: "2" },
                { label: "3" },
                { label: "4" },
              ],
            },
          ],
        },
      ],
    });
  });
});
