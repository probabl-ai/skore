import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";
import FileTree from "../FileTree.vue";

describe("FileTree", () => {
  it("renders properly", () => {
    const wrapper = mount(FileTree);
    expect(wrapper.text()).toContain("file tree");
  });
});
