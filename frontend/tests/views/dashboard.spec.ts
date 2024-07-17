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
        slug: ["a", "b"],
      },
    }));

    const wrapper = await mountSuspense(DashboardView);
    const fileTree = wrapper.getComponent(FileTree);
    expect(fileTree.props()).toEqual({
      nodes: [
        {
          path: "probabl-ai",
          children: [
            {
              path: "probabl-ai/demo-usecase",
              children: [
                {
                  path: "probabl-ai/demo-usecase/training",
                  children: [{ path: "probabl-ai/demo-usecase/training/0" }],
                },
              ],
            },
            {
              path: "probabl-ai/test-mandr",
              children: [
                { path: "probabl-ai/test-mandr/0" },
                { path: "probabl-ai/test-mandr/1" },
                { path: "probabl-ai/test-mandr/2" },
                { path: "probabl-ai/test-mandr/3" },
                { path: "probabl-ai/test-mandr/4" },
              ],
            },
          ],
        },
      ],
    });
  });
});
