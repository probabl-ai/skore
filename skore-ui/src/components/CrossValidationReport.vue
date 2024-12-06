<script lang="ts">
export interface ScalarResult {
  name: string;
  value: number;
  fold?: number;
  label?: string;
}

export interface TabularResult {
  name: string;
  columns: any[];
  data: any[][];
  favorability: "higher-is-better" | "lower-is-better";
}

export interface PrimaryResults {
  scalarResults: ScalarResult[];
  tabularResults: TabularResult[];
}
</script>

<script setup lang="ts">
import CrossValidationReportResults from "@/components/CrossValidationReportResults.vue";
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";

const fake: PrimaryResults = {
  scalarResults: [
    { name: "toto", value: 4.32 },
    { name: "tata", value: 4.32 },
    { name: "titi", value: 4.32, fold: 1 },
    { name: "tutu", value: 4.32 },
    { name: "stab", value: 0.002, label: "Good" },
  ],
  tabularResults: [
    {
      name: "a",
      columns: ["Accuracy", "Precision", "Recall", "F1 Score"],
      data: [
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
      ],
      favorability: "higher-is-better",
    },
    {
      name: "b",
      columns: ["Accuracy", "Precision", "Recall", "F1 Score"],
      data: [
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 0.4, 0.5, 0.6],
      ],
      favorability: "lower-is-better",
    },
  ],
};
</script>

<template>
  <div class="cross-validation-report">
    <TabPanel>
      <TabPanelContent name="Primary Results" icon="icon-bar-chart">
        <CrossValidationReportResults
          :scalar-results="fake.scalarResults"
          :tabular-results="fake.tabularResults"
        />
      </TabPanelContent>
      <TabPanelContent name="Additional plots" icon="icon-large-bar-chart"></TabPanelContent>
      <TabPanelContent name="Storage/Details" icon="icon-hard-drive"></TabPanelContent>
    </TabPanel>
  </div>
</template>
