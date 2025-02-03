<script lang="ts">
export enum Favorability {
  GREATER_IS_BETTER = "greater_is_better",
  LOWER_IS_BETTER = "lower_is_better",
  UNKNOWN = "unknown",
}

export interface EstimatorReportScalarResult {
  name: string;
  value: number;
  stddev?: number;
  label?: string;
  description?: string;
  favorability: Favorability;
}

export interface EstimatorReportTabularResult {
  name: string;
  columns: any[];
  data: any[][];
  favorability: Favorability[];
}

export interface EstimatorReportPrimaryResults {
  scalarResults: EstimatorReportScalarResult[];
  tabularResults: EstimatorReportTabularResult[];
}

export interface EstimatorReportPlot {
  name: string;
  value: any;
}

export interface EstimatorReportDetailSectionItem {
  name: string;
  description: string;
  value: string;
}

export interface EstimatorReportDetailSection {
  title: string;
  icon: string;
  items: EstimatorReportDetailSectionItem[];
}
</script>

<script setup lang="ts">
import EstimatorReportDetails from "@/components/EstimatorReportDetails.vue";
import EstimatorReportPlots from "@/components/EstimatorReportPlots.vue";
import EstimatorReportResults from "@/components/EstimatorReportResults.vue";
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";

const props = defineProps<{
  scalarResults: EstimatorReportScalarResult[];
  tabularResults: EstimatorReportTabularResult[];
  plots: EstimatorReportPlot[];
  sections: EstimatorReportDetailSection[];
}>();
</script>

<template>
  <div class="cross-validation-report">
    <TabPanel>
      <TabPanelContent name="Primary Results" icon="icon-bar-chart">
        <EstimatorReportResults
          :scalar-results="props.scalarResults"
          :tabular-results="props.tabularResults"
        />
      </TabPanelContent>
      <TabPanelContent name="Plots" icon="icon-large-bar-chart">
        <EstimatorReportPlots :plots="props.plots" />
      </TabPanelContent>
      <TabPanelContent name="Storage/Details" icon="icon-hard-drive">
        <EstimatorReportDetails :sections="props.sections" />
      </TabPanelContent>
    </TabPanel>
  </div>
</template>
