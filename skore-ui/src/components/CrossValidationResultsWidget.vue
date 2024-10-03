<script setup lang="ts">
import type { VisualizationSpec } from "vega-embed";
import type { DataFrameWidgetProps } from "@/components/DataFrameWidget.vue";
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";

const props = defineProps<{
  roc_curve_spec: VisualizationSpec;
  cv_results_table: DataFrameWidgetProps;
}>();
</script>

<template>
  <div class="content">
    <VegaWidget :spec="props.roc_curve_spec" />
    <DataFrameWidget
      :columns="props.cv_results_table.columns"
      :data="props.cv_results_table.data"
    />
  </div>
</template>

<style scoped>
div.content {
  display: flex;
}
</style>
