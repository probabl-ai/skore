import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import ComponentsView from "@/views/ComponentsView.vue";

describe("ComponentsView", () => {
  it("Can render.", async () => {
    const wrapper = mount(ComponentsView);
    expect(wrapper.html().length).toBeGreaterThan(0);
  });
});
