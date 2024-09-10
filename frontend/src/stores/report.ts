import { defineStore } from "pinia";
import { ref, shallowRef, toRaw } from "vue";

import { type KeyLayoutSize, type KeyMoveDirection, type Layout, type ReportItem } from "@/models";
import { fetchReport } from "@/services/api";
import { poll, sha1, swapItemsInArray } from "@/services/utils";

export const useReportStore = defineStore("reports", () => {
  // this object is not deeply reactive as it may be very large
  const report = shallowRef<{ [key: string]: ReportItem } | null>(null);
  const layout = ref<Layout>([]);

  /**
   * Return true if the the given key is in the list of displayed keys, false otherwise.
   * @param key the key to look for
   */
  function isKeyDisplayed(key: string) {
    const visibleKeys = layout.value.map(({ key: k }) => k);
    return visibleKeys.includes(key);
  }

  /**
   * Add the value of a key to the report.
   * @param key the key to add to the report
   */
  async function displayKey(key: string) {
    stopBackendPolling();
    if (!isKeyDisplayed(key)) {
      layout.value.push({ key, size: "large" });
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Hide the value of a key from the report.
   * @param key the key to hide
   */
  async function hideKey(key: string) {
    stopBackendPolling();
    if (isKeyDisplayed(key)) {
      layout.value = layout.value.filter(({ key: k }) => key != k);
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Change the visual size of the value of a key in the report.
   * @param key the key which changes
   * @param size the target size
   */
  async function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    stopBackendPolling();
    if (isKeyDisplayed(key)) {
      const index = layout.value.findIndex(({ key: k }) => key == k);
      if (index !== -1) {
        layout.value[index].size = size;
      }
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Move a displayed key up or down in the list
   * @param key the key to replace
   * @param direction up or down
   */
  async function moveKey(key: string, direction: KeyMoveDirection) {
    stopBackendPolling();
    const offset = direction == "up" ? -1 : 1;
    const index = layout.value.findIndex(({ key: k }) => key == k);
    swapItemsInArray(layout.value, index, index + offset);
    await persistLayout();
    await startBackendPolling();
  }

  /**
   * Fetch all reports URI
   * and eventually the detail of the currently selected report
   */
  let _isCanceledCall = false;
  async function fetch() {
    if (!_isCanceledCall) {
      const r = await fetchReport();
      if (r) {
        setReport(r);
      }
    }
  }
  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: Function | null = null;
  async function startBackendPolling() {
    _isCanceledCall = false;
    _stopBackendPolling = await poll(fetch, 1500);
  }

  /**
   * Stop real time sync with the server.
   */
  function stopBackendPolling() {
    _isCanceledCall = true;
    _stopBackendPolling && _stopBackendPolling();
  }

  /**
   * Send new layout to backend
   * ⚠️ temporary save to the local storage
   * until backend accepts to store layouts
   */
  async function persistLayout() {
    if (report.value && layout.value) {
      const rawReport = toRaw(report.value);
      const rawLayout = toRaw(layout.value);
      const storageKey = await sha1(JSON.stringify(rawReport));
      localStorage.setItem(storageKey, JSON.stringify(rawLayout));
    }
  }

  /**
   * Set the current report and populate the layout
   * @param r data received from the backend
   */
  async function setReport(r: { [key: string]: ReportItem }) {
    report.value = r;
    const rawReport = toRaw(report.value);
    const storageKey = await sha1(JSON.stringify(rawReport));
    const l = localStorage.getItem(storageKey);
    if (l !== null) {
      layout.value = JSON.parse(l);
    }
  }

  return {
    layout,
    report,
    displayKey,
    hideKey,
    setKeyLayoutSize,
    startBackendPolling,
    stopBackendPolling,
    setReport,
    moveKey,
  };
});
