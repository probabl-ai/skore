import { VueWrapper } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { useRoute } from "vue-router";

import DashboardView from "@/views/DashboardView.vue";

import FileTree from "@/components/FileTree.vue";
import { beforeEach } from "node:test";
import { mountSuspense } from "../test.utils";

beforeEach(() => {
  vi.mock("vue-router", () => ({
    useRoute: vi.fn(),
    useRouter: vi.fn(() => ({
      push: () => {},
    })),
  }));
});

describe("DashboardView", () => {
  it("Renders properly", async () => {
    (useRoute as any).mockImplementationOnce(() => ({
      params: {
        segments: ["a", "b"],
      },
    }));

    const dashboard = await mountSuspense(DashboardView);
    // i.e. not a `VueError`
    expect(dashboard).toBeInstanceOf(VueWrapper);

    const fileTree = await dashboard.findComponent(FileTree);
    expect(fileTree).toBeInstanceOf(VueWrapper);
  });
});
