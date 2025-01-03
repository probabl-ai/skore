<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref, toValue, watch } from "vue";

import TextInput from "@/components/TextInput.vue";

export interface DataFrameWidgetProps {
  index: any[];
  columns: any[];
  data: any[][];
  indexNames: any[];
}
const props = defineProps<DataFrameWidgetProps>();

const rowPerPage = ref(10);
const currentPage = ref(0);
const search = defineModel<string>("search");

const rows = computed(() => {
  let filteredRowIndexes: number[];
  if (search.value !== undefined && search.value.length > 0) {
    const searchToken = search.value.toLowerCase();
    filteredRowIndexes = props.data.reduce((acc, row, i) => {
      let index = props.index[i];
      if (index instanceof Array) {
        index = index.join(" ");
      } else {
        index = index.toString();
      }
      const text = row.join(" ").toLowerCase();
      if (text.includes(searchToken) || index.includes(searchToken)) {
        acc.push(i);
      }
      return acc;
    }, []);
  } else {
    filteredRowIndexes = props.data.map((_, i) => i);
  }

  return filteredRowIndexes.map((i) => {
    const index = props.index[i];
    if (index instanceof Array) {
      return [index.join(", "), ...props.data[i]];
    }
    return [index, ...props.data[i]];
  });
});

const totalPages = computed(() => {
  return Math.ceil(rows.value.length / rowPerPage.value);
});

const pageStart = computed(() => {
  return currentPage.value * rowPerPage.value;
});

const pageEnd = computed(() => {
  return (currentPage.value + 1) * rowPerPage.value;
});

const visibleRows = computed(() => {
  return rows.value.slice(pageStart.value, pageEnd.value);
});

const indexNamesColSpan = computed(() => {
  const hasIndexNames = props.indexNames.some((name) => name !== null);
  if (hasIndexNames) {
    return props.indexNames.length;
  }
  return 1;
});

function nextPage() {
  if (currentPage.value < totalPages.value - 1) {
    currentPage.value++;
  }
}

function previousPage() {
  if (currentPage.value > 0) {
    currentPage.value--;
  }
}

function onPageSizeChange(event: Event) {
  currentPage.value = 0;
  rowPerPage.value = parseInt((event.target as HTMLSelectElement).value);
}

watch(
  [() => toValue(props.data), () => toValue(props.columns)],
  ([newData, newColumns], [oldData, oldColumns]) => {
    if (newData.length != oldData.length || newColumns.length != oldColumns.length) {
      currentPage.value = 0;
    }
  }
);
</script>

<template>
  <div class="data-frame-widget">
    <TextInput v-model="search" icon="icon-search" placeholder="Search" class="search-input" />
    <Simplebar>
      <table>
        <thead>
          <tr>
            <th :colspan="indexNamesColSpan">index</th>
            <th v-for="(name, i) in props.columns" :key="i">{{ name }}</th>
          </tr>
          <tr v-if="indexNamesColSpan > 1">
            <th v-for="(name, i) in props.indexNames" :key="i" class="named-index">
              {{ name }}
            </th>
            <th v-for="(name, i) in props.columns" :key="i"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in visibleRows" :key="i">
            <template v-if="indexNamesColSpan === 1">
              <td v-for="(value, i) in row" :key="i" :colspan="i === 0 ? indexNamesColSpan : 1">
                {{ value }}
              </td>
            </template>
            <template v-else>
              <td v-for="(value, i) in row[0].split(', ')" :key="i" class="named-index">
                {{ value }}
              </td>
              <td v-for="(value, i) in row.slice(1)" :key="i">
                {{ value }}
              </td>
            </template>
          </tr>
        </tbody>
      </table>
    </Simplebar>
    <div class="pagination" v-if="rows.length > 10">
      <div class="pagination-page-size">
        Page size
        <select @change="onPageSizeChange">
          <option value="10">10</option>
          <option value="25">25</option>
          <option value="50">50</option>
        </select>
      </div>
      <div class="pagination-buttons" v-if="totalPages > 1">
        <button @click="currentPage = 0" :disabled="currentPage == 0">&lt;&lt;</button>
        <button @click="previousPage" :disabled="currentPage == 0">&lt;</button>
        <button @click="nextPage" :disabled="currentPage == totalPages - 1">&gt;</button>
        <button @click="currentPage = totalPages - 1" :disabled="currentPage == totalPages - 1">
          &gt;&gt;
        </button>
      </div>
      <div class="page-info">
        Results: {{ pageStart + 1 }}-{{ Math.min(pageEnd, rows.length) }} of {{ rows.length }}
        <span v-if="search && search.length > 0">
          (filtered from {{ props.data.length }} results)</span
        >
      </div>
    </div>
  </div>
</template>

<style scoped>
.data-frame-widget {
  max-width: 100%;

  .search-input {
    max-width: 100%;
  }

  table {
    min-width: 100%;
    border: 1px var(--stroke-width-md) var(--color-stroke-background-primary);
    border-radius: var(--radius-xs);
    margin-top: var(--spacing-4);
    border-collapse: separate;
    border-spacing: 0;
    overflow-x: auto;
    text-align: right;

    & thead {
      background-color: var(--color-background-secondary);

      & tr {
        & th {
          padding: var(--spacing-4);
          color: var(--color-text-primary);
          font-size: var(--font-size-sm);
          font-weight: var(--font-weight-medium);
          text-align: right;

          &:first-child,
          &.named-index {
            background-color: var(--color-background-secondary);
          }

          &[colspan="1"]:first-child:not(.named-index) {
            position: sticky;
            left: 0;
          }
        }
      }
    }

    & tbody {
      & tr {
        position: relative;
        padding: var(--spacing-4);

        & td {
          padding: var(--spacing-4);

          &:first-child,
          &.named-index {
            background-color: var(--color-background-secondary);
            text-align: left;
          }

          &:first-child:not(.named-index) {
            position: sticky;
            left: 0;

            &::after {
              position: absolute;
              top: 0;
              right: -3px;
              width: 3px;
              height: 100%;
              background: linear-gradient(to right, var(--color-background-secondary), transparent);
              content: " ";
            }
          }
        }

        &:last-child {
          border-bottom: none;
        }
      }
    }

    /* stylelint-disable no-descending-specificity */
    & > thead > tr:not(:last-child) > th,
    & > thead > tr:not(:last-child) > td,
    & > tbody > tr:not(:last-child) > th,
    & > tbody > tr:not(:last-child) > td,
    & > tfoot > tr:not(:last-child) > th,
    & > tfoot > tr:not(:last-child) > td,
    & > thead:not(:last-child),
    & > tbody:not(:last-child),
    & > tfoot:not(:last-child) {
      border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
    }
  }

  .pagination {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: var(--spacing-8);
    color: var(--color-text-primary);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-regular);

    .pagination-buttons {
      display: flex;
      align-items: center;
      gap: var(--spacing-8);

      & button {
        border: none;
        background-color: transparent;
        color: var(--color-text-primary);
        cursor: pointer;
        font-size: var(--font-size-sm);
        font-weight: var(--font-weight-regular);

        &:disabled,
        &[disabled] {
          color: var(--color-text-secondary);
        }
      }
    }

    & select {
      padding: var(--spacing-4);
      border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
      border-radius: var(--radius-xs);
      margin-left: var(--spacing-8);
      background-color: var(--color-background-secondary);
      box-shadow: 0 1px 2px var(--color-shadow);
      color: var(--color-text-primary);
      cursor: pointer;
    }
  }
}
</style>
