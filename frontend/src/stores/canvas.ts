import { debounce } from "lodash";
import { defineStore } from "pinia";
import { ref, shallowRef } from "vue";

import { DataStore, type KeyLayoutSize, type Layout } from "@/models";
import { putLayout } from "@/services/api";

export const useCanvasStore = defineStore("canvas", () => {
  // this object is not deeply reactive as it may be very large
  const dataStore = shallowRef<DataStore | null>(null);
  const layout = ref<Layout>([]);

  async function displayKey(key: string) {
    layout.value.push({ key, size: "large" });
    await syncLayout();
  }

  async function hideKey(key: string) {
    layout.value = layout.value.filter(({ key: k }) => key != k);
    await syncLayout();
  }

  async function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    const index = layout.value.findIndex(({ key: k }) => key == k);
    if (index !== -1) {
      layout.value[index].size = size;
    }
    await syncLayout();
  }

  async function _syncLayout() {
    // Send new layout to backend
    if (dataStore.value && layout.value) {
      const refreshed = await putLayout(dataStore.value.uri, layout.value);
      if (refreshed) {
        dataStore.value = refreshed;
      }
    }
  }
  const syncLayout = debounce(_syncLayout, 1000, { leading: true, trailing: false });

  function setDataStore(ds: DataStore | null) {
    dataStore.value = ds;
    if (ds?.layout) {
      layout.value = ds.layout;
    }
  }

  return {
    dataStore,
    layout,
    displayKey,
    hideKey,
    setDataStore,
    setKeyLayoutSize,
  };
});
