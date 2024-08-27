import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import DashboardView from "@/views/DashboardView.vue";

import { mountSuspense } from "../test.utils";

const app = createApp({});

describe("DashboardView", () => {
  beforeEach(() => {
    vi.mock("vue-router", () => ({
      useRoute: vi.fn(),
      useRouter: vi.fn(() => ({
        push: () => {},
      })),
    }));
    const pinia = createPinia();
    app.use(pinia);
    setActivePinia(pinia);
  });

  it("Renders properly", async () => {
    (useRoute as any).mockImplementationOnce(() => ({
      params: {
        segments: ["a", "b"],
      },
    }));

    const dashboard = await mountSuspense(DashboardView);
    // i.e. not a `VueError`
    expect(dashboard).toBeInstanceOf(VueWrapper);
  });
});
