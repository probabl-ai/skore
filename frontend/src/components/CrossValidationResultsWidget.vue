<script setup lang="ts">
import type { VisualizationSpec } from "vega-embed";
import type { DataFrameWidgetProps } from "@/components/DataFrameWidget.vue";
import DataFrameWidget from "@/components/DataFrameWidget.vue";
import VegaWidget from "@/components/VegaWidget.vue";

const props = defineProps<{
  test_score_plot: VisualizationSpec;
  cv_results_table: DataFrameWidgetProps;
}>();
</script>

<template>
  <div class="content">
    <VegaWidget :spec="props.test_score_plot" />
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
