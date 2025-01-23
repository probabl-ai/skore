import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

describe("Icons", () => {
  it("Can render.", async () => {
    interface ModuleImportInterface {
      default: never;
    }
    const iconModules = import.meta.glob<ModuleImportInterface>("@/components/icons/**/*.vue");
    for (const path in iconModules) {
      const M = await iconModules[path]();
      const wrapper = mount(M.default);
      expect(wrapper.html().length).toBeGreaterThan(0);
    }
  });
});
