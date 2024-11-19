<script setup lang="ts">
import "@finos/perspective-viewer";
import "@finos/perspective-viewer-d3fc";
import "@finos/perspective-viewer-datagrid";

import { getPerspectiveWorker } from "@/workers";
import { type Table } from "@finos/perspective";
import { type HTMLPerspectiveViewerElement } from "@finos/perspective-viewer";
import { onMounted, onUnmounted, useTemplateRef } from "vue";

const props = defineProps<{ data: Record<string, unknown>[] }>();
const viewer = useTemplateRef<HTMLPerspectiveViewerElement>("viewer");
let table: Table;

async function setupPerspective() {
  const worker = await getPerspectiveWorker();
  table = await worker.table(props.data);
  if (viewer.value) {
    await viewer.value.load(table);
    viewer.value.restore({ settings: true });
  }
}

onMounted(async () => {
  await setupPerspective();
});

onUnmounted(async () => {
  if (table) {
    table.free();
  }
});
</script>

<template>
  <div class="perspective">
    <perspective-viewer ref="viewer"></perspective-viewer>
  </div>
</template>

<style>
@import url("@finos/perspective-styles/intl.css");
@import url("@finos/perspective-styles/icons.css");
@import url("@finos/perspective-styles/pro.css");

perspective-viewer {
  width: 100%;
  height: 500px;
}
</style>
