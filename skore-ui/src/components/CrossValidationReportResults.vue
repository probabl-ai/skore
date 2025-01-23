<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref, useTemplateRef } from "vue";

import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import FloatingTooltip from "@/components/FloatingTooltip.vue";
import MetricFavorability from "@/components/MetricFavorability.vue";
import StaticRange from "@/components/StaticRange.vue";
import type { PrimaryResultsDto, TabularResultDto } from "@/dto";
import { isElementOverflowing } from "@/services/utils";

const props = defineProps<PrimaryResultsDto>();
const currentTabularResultIndex = ref(0);
const currentTabularResult = computed<TabularResultDto>(() => {
  return props.tabularResults[currentTabularResultIndex.value];
});
const scalarResultsDivs = useTemplateRef<HTMLDivElement[]>("scalarResultsDivs");

function exponential(x: number) {
  if (typeof x !== "number") {
    return x;
  }
  if (x >= 0.1 && x <= 999) {
    return x.toFixed(4);
  }
  return x.toExponential(2);
}

function isNameTooltipEnabled(index: number) {
  if (scalarResultsDivs.value) {
    const name = scalarResultsDivs.value[index].querySelector(".name");
    if (name) {
      return isElementOverflowing(name);
    }
  }
  return false;
}
</script>

<template>
  <div class="cross-validation-report-result">
    <div class="scalar-results">
      <div
        v-for="(result, i) in props.scalarResults"
        :key="i"
        class="result"
        ref="scalarResultsDivs"
      >
        <FloatingTooltip placement="bottom" :enabled="isNameTooltipEnabled(i)">
          <div class="name">
            {{ result.name }}
            <MetricFavorability :favorability="result.favorability" />
          </div>
          <template #tooltipContent>
            <span class="name-tooltip">{{ result.name }}</span>
          </template>
        </FloatingTooltip>
        <div class="labeled-value" v-if="result.label">
          <div class="label">{{ result.label }}</div>
          <StaticRange :value="result.value" />
          <div class="description">{{ result.description }}</div>
        </div>
        <div class="value" v-else>
          <FloatingTooltip placement="bottom-start">
            <div>{{ exponential(result.value) }}</div>
            <div v-if="result.stddev" class="fold">± {{ exponential(result.stddev) }}</div>
            <template #tooltipContent>
              <div class="value-tooltip">
                <div>Mean: {{ result.value }}</div>
                <div v-if="result.stddev">Std dev: {{ result.stddev }}</div>
              </div>
            </template>
          </FloatingTooltip>
        </div>
      </div>
    </div>
    <div class="tabular-results" v-if="currentTabularResult">
      <div class="header">
        <div class="name">
          <i class="icon icon-large-bar-chart" /> {{ currentTabularResult.name }}
        </div>
        <DropdownButton
          v-if="props.tabularResults.length > 1"
          :label="currentTabularResult.name"
          icon="icon-chevron-down"
          align="left"
          icon-position="right"
        >
          <Simplebar>
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
        <Simplebar>
          <table>
            <thead>
              <tr>
                <th>Fold</th>
                <th v-for="(column, i) in currentTabularResult.columns" :key="i">
                  {{ column }}
                  <MetricFavorability :favorability="currentTabularResult.favorability[i]" />
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in currentTabularResult.data" :key="i">
                <td>Fold&nbsp;{{ i + 1 }}</td>
                <td v-for="(value, j) in row" :key="j">
                  <FloatingTooltip placement="left">
                    {{ exponential(value) }}
                    <template #tooltipContent>
                      <span class="value-tooltip">{{ value }}</span>
                    </template>
                  </FloatingTooltip>
                </td>
              </tr>
            </tbody>
          </table>
        </Simplebar>
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
    grid-template-columns: repeat(var(--nb-scalar-columns, 2), 1fr);

    &:last-child {
      border-right: none;
    }

    & .result {
      padding: var(--spacing-8) var(--spacing-10);
      background-color: var(--color-background-primary);

      & .name {
        overflow: hidden;
        color: var(--color-text-secondary);
        font-size: var(--font-size-xs);
        text-overflow: ellipsis;
        white-space: nowrap;
        word-break: break-all;

        & .name-tooltip {
          font-size: var(--font-size-xxs);
        }
      }

      & .labeled-value {
        color: var(--color-text-primary);
        font-size: var(--font-size-xlg);

        & .description {
          color: var(--color-text-secondary);
          font-size: var(--font-size-xxs);
        }
      }

      & .value {
        color: var(--color-text-primary);
        font-size: var(--font-size-xlg);

        & .value-tooltip {
          color: var(--color-text-primary);
          font-size: var(--font-size-xs);
        }

        & .fold {
          color: var(--color-text-primary);
          font-size: var(--font-size-xs);
        }
      }

      & .icon {
        color: var(--color-icon-tertiary);
        vertical-align: middle;
      }
    }
  }

  & .tabular-results {
    max-width: 100%;

    & .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: var(--spacing-16) var(--spacing-10);

      & .name {
        overflow: hidden;
        color: var(--color-text-primary);
        font-size: var(--font-size-sm);
        text-overflow: ellipsis;
        white-space: nowrap;

        & .icon {
          color: var(--color-text-branding);
          vertical-align: middle;
        }
      }

      & .dropdown {
        flex: 0 1 40%;

        & button {
          padding: var(--spacing-6) var(--spacing-10);

          & .label {
            overflow: hidden;
            text-overflow: ellipsis;
          }
        }
      }
    }

    & .result {
      overflow: hidden;
      max-width: 100%;

      & table {
        --fold-column-width: 70px;

        min-width: 100%;
        border-collapse: collapse;
        text-align: right;

        & thead tr th {
          padding: var(--spacing-6) var(--spacing-10);
          border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
          border-bottom-color: var(--color-background-primary);
          background-color: var(--color-background-secondary);
          color: var(--color-text-primary);
          font-weight: var(--font-weight-medium);

          &:first-child {
            position: sticky;
            left: 0;
            width: var(--fold-column-width);
            border-left: none;
            text-align: center;
          }

          &:last-child {
            border-right: none;
          }
        }

        & tbody tr {
          position: relative;
          color: var(--color-text-primary);
          font-weight: var(--font-weight-regular);

          & td {
            padding: var(--spacing-6) var(--spacing-10);
            border: solid var(--stroke-width-md) var(--color-stroke-background-primary);

            &:first-child {
              position: sticky;
              z-index: 2;
              left: 0;
              width: var(--fold-column-width);
              border-bottom-color: var(--color-background-primary);
              border-left: none;
              background-color: var(--color-background-secondary);
              font-weight: var(--font-weight-medium);
              text-align: left;

              &::after {
                position: absolute;
                top: 0;
                right: -3px;
                width: 3px;
                height: 100%;
                background: linear-gradient(
                  to right,
                  var(--color-background-secondary),
                  transparent
                );
                content: " ";
              }
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
}

@media screen and (width >= 801px) {
  .cross-validation-report-result {
    --nb-scalar-columns: 3;
  }
}

@media screen and (width >= 1025px) {
  .cross-validation-report-result {
    --nb-scalar-columns: 5;
  }
}
</style>
