import { mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ComponentsView from "@/views/ComponentsView.vue";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";

describe("ComponentsView", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can render.", async () => {
    const wrapper = mount(ComponentsView);
    expect(wrapper.html().length).toBeGreaterThan(0);
  });
});
