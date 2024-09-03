import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import ReportBuilderView from "@/views/ReportBuilderView.vue";

import { mountSuspense } from "../test.utils";

const app = createApp({});

describe("ReportBuilderView", () => {
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

    const builder = await mountSuspense(ReportBuilderView);
    // i.e. not a `VueError`
    expect(builder).toBeInstanceOf(VueWrapper);
  });
});
