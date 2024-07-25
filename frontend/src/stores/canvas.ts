import { defineStore } from "pinia";
import { ref } from "vue";

import { type DataStore } from "@/models";

export type KeyLayoutSize = "small" | "medium" | "large";

export const useCanvasStore = defineStore("canvas", () => {
  const dataStore = ref<DataStore | null>();
  const displayedKeys = ref<string[]>([]);
  const layoutSizes = ref<{ [key: string]: KeyLayoutSize }>({});

  function displayKey(key: string) {
    if (displayedKeys.value.indexOf(key) === -1) {
      displayedKeys.value.push(key);
    }
  }

  function hideKey(key: string) {
    displayedKeys.value = displayedKeys.value.filter((k) => k != key);
  }

  function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    layoutSizes.value[key] = size;
  }

  function setDataStore(ds: DataStore | null) {
    dataStore.value = ds;
    displayedKeys.value = [];
    layoutSizes.value = {};
  }

  function get(key: string) {
    // Temp function to access a key anywhere in the data store
    const ds = dataStore.value;
    if (ds) {
      if (key in ds.views) {
        return ds.views[key];
      }
      if (key in ds.logs) {
        return ds.logs[key];
      }
      if (key in ds.artifacts) {
        return ds.artifacts[key];
      }
      if (key in ds.info) {
        return ds.info[key];
      }
    }
  }

  return { displayedKeys, layoutSizes, displayKey, hideKey, setKeyLayoutSize, setDataStore, get };
});
