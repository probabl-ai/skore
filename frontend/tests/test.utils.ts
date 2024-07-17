import { flushPromises, mount } from "@vue/test-utils";
import { vi } from "vitest";
import { type ComponentPublicInstance, defineComponent, h, Suspense } from "vue";

export const mockedFecth = vi.fn();
global.fetch = mockedFecth;

export function createFetchResponse(data: object, status = 200) {
  return { json: () => new Promise((resolve) => resolve(data)), status };
}

export async function mountSuspens(
  component: new () => ComponentPublicInstance,
  options: any = {}
) {
  const suspensedComponent = defineComponent({
    render: () => {
      return h(Suspense, null, {
        default: h(component),
        fallback: h("div", "loading..."),
      });
    },
  });
  const wrapper = mount(suspensedComponent, options);
  await flushPromises();
  return wrapper;
}
