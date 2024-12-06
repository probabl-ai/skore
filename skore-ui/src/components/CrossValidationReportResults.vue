<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref } from "vue";

import type { PrimaryResults, TabularResult } from "@/components/CrossValidationReport.vue";
import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";

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
          <span v-if="result.fold" class="fold">(Fold {{ result.fold + 1 }})</span>
        </div>
      </div>
    </div>
    <div class="tabular-results">
      <div class="header">
        <div class="name">
          <i class="icon icon-large-bar-chart" /> {{ currentTabularResult.name }}
        </div>
        <DropdownButton
          icon="icon-chevron-down"
          align="left"
          icon-position="right"
          :label="currentTabularResult.name"
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
      </div>
      <div class="result">
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

<style>
.cross-validation-report-result {
  & .scalar-results {
    display: grid;
    flex-direction: row;
    border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    background-color: var(--color-stroke-background-primary);
    gap: 1px;
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr;

    &:last-child {
      border-right: none;
    }

    & .result {
      padding: var(--spacing-8) var(--spacing-10);
      background-color: var(--color-background-primary);

      & .name {
        color: var(--color-text-secondary);
        font-size: var(--font-size-xs);
      }

      & .value {
        color: var(--color-text-primary);
        font-size: var(--font-size-xlg);

        & .fold {
          color: var(--color-background-branding);
          font-size: var(--font-size-xs);
        }
      }
    }
  }

  & .tabular-results {
    & .header {
      display: flex;
      justify-content: space-between;
      padding: var(--spacing-16) var(--spacing-10);

      & .name {
        color: var(--color-text-primary);
        font-size: var(--font-size-sm);

        & .icon {
          color: var(--color-text-branding);
          vertical-align: middle;
        }
      }
    }

    & table {
      --fold-column-width: 70px;

      width: 100%;
      border-collapse: collapse;
      text-align: right;

      & thead tr th {
        padding: var(--spacing-4);
        border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
        border-bottom-color: var(--color-background-primary);
        background-color: var(--color-background-secondary);
        color: var(--color-text-primary);
        font-weight: var(--font-weight-medium);

        &:first-child {
          width: var(--fold-column-width);
          border-left: none;
          text-align: left;
        }

        &:last-child {
          border-right: none;
        }
      }

      & tbody tr {
        color: var(--color-text-primary);
        font-weight: var(--font-weight-regular);

        & td {
          padding: var(--spacing-4);
          border: solid var(--stroke-width-md) var(--color-stroke-background-primary);

          &:first-child {
            width: var(--fold-column-width);
            border-bottom-color: var(--color-background-primary);
            border-left: none;
            background-color: var(--color-background-secondary);
            font-weight: var(--font-weight-medium);
            text-align: left;
          }

          &:last-child {
            border-right: none;
          }
        }

        &:last-child {
          & td {
            border-bottom: none;
            border-bottom-left-radius: var(--radius-xs);
          }
        }
      }
    }
  }
}
</style>
