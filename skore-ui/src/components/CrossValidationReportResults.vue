<script setup lang="ts">
import Simplebar from "simplebar-vue";

import type { PrimaryResults, TabularResult } from "@/components/CrossValidationReport.vue";
import { computed, ref } from "vue";
import DropdownButton from "./DropdownButton.vue";
import DropdownButtonItem from "./DropdownButtonItem.vue";

const props = defineProps<PrimaryResults>();
const currentTabularResultIndex = ref(0);
const currentTabularResult = computed<TabularResult>(() => {
  return props.tabularResults[currentTabularResultIndex.value];
});
</script>

<template>
  <div class="cross-validation-report-result">
    <div class="scalar-results">
      <div v-for="(result, i) in props.scalarResults" :key="i" class="result">
        <div class="name">{{ result.name }}</div>
        <div class="value">
          {{ result.value }}
          <span v-if="result.fold" class="fold">{{ result.fold }}</span>
        </div>
      </div>
    </div>
    <div class="tabular-results">
      <DropdownButton
        icon="icon-chevron-down"
        :label="currentTabularResult.name"
        align="right"
        :is-primary="true"
      >
        <Simplebar class="history-items">
          <DropdownButtonItem
            v-for="(result, i) in props.tabularResults"
            :key="i"
            :label="result.name"
            @click="currentTabularResultIndex = i"
          />
        </Simplebar>
      </DropdownButton>
      <div class="result">
        <div class="name">{{ currentTabularResult.name }}</div>
        <table>
          <thead>
            <tr>
              <th>Fold</th>
              <th v-for="(column, i) in currentTabularResult.columns" :key="i">{{ column }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in currentTabularResult.data" :key="i">
              <td>Fold {{ i + 1 }}</td>
              <td v-for="(value, j) in row" :key="j">{{ value }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
