import FileTree from "@/components/FileTree.vue";
import DashboardView from "@/views/DashboardView.vue";
import { describe, expect, it, vi } from "vitest";
import { useRoute } from "vue-router";
import { createFetchResponse, mockedFetch, mountSuspense } from "../test.utils";

vi.mock("vue-router", () => ({
  useRoute: vi.fn(),
  useRouter: vi.fn(() => ({
    push: () => {},
  })),
}));

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

    (useRoute as any).mockImplementationOnce(() => ({
      params: {
        segments: ["a", "b"],
      },
    }));

    const wrapper = await mountSuspense(DashboardView);
    const fileTree = wrapper.getComponent(FileTree);
    expect(fileTree.props()).toEqual({
      nodes: [
        {
          uri: "probabl-ai",
          children: [
            {
              uri: "probabl-ai/demo-usecase",
              children: [
                {
                  uri: "probabl-ai/demo-usecase/training",
                  children: [{ uri: "probabl-ai/demo-usecase/training/0" }],
                },
              ],
            },
            {
              uri: "probabl-ai/test-mandr",
              children: [
                { uri: "probabl-ai/test-mandr/0" },
                { uri: "probabl-ai/test-mandr/1" },
                { uri: "probabl-ai/test-mandr/2" },
                { uri: "probabl-ai/test-mandr/3" },
                { uri: "probabl-ai/test-mandr/4" },
              ],
            },
          ],
        },
      ],
    });
  });
});
