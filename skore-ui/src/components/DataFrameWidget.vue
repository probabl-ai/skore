<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref, toValue, watch } from "vue";

import TextInput from "@/components/TextInput.vue";

export interface DataFrameWidgetProps {
  index: any[];
  columns: any[];
  data: any[][];
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

watch([() => toValue(props.data), () => toValue(props.columns)], () => {
  currentPage.value = 0;
});
</script>

<template>
  <div class="data-frame-widget">
    <TextInput v-model="search" icon="icon-search" placeholder="Search" class="search-input" />
    <Simplebar>
      <table>
        <thead>
          <tr>
            <th>index</th>
            <th v-for="(name, i) in props.columns" :key="i">{{ name }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in visibleRows" :key="i">
            <td v-for="(value, i) in row" :key="i">{{ value }}</td>
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
    border: 1px solid var(--border-color-normal);
    border-radius: var(--border-radius);
    margin-top: var(--spacing-gap-small);
    border-collapse: separate;
    border-spacing: 0;
    overflow-x: auto;
    text-align: right;

    & thead {
      background-color: var(--background-color-elevated);
      color: var(--text-color-normal);
      font-size: var(--text-size-normal);
      font-weight: var(--text-weight-normal);

      & tr {
        & th {
          padding: var(--spacing-padding-small);

          &:first-child {
            position: sticky;
            left: 0;
            background-color: var(--background-color-elevated);
            text-align: left;
          }
        }
      }
    }

    & tbody {
      & tr {
        padding: var(--spacing-padding-small);

        & td {
          padding: var(--spacing-padding-small);
          color: var(--text-color-highlight);
          font-size: var(--text-size-highlight);
          font-weight: var(--text-weight-highlight);

          &:first-child {
            position: sticky;
            left: 0;
            width: auto;
            background-color: var(--background-color-elevated);
            color: var(--text-color-normal);
            font-size: var(--text-size-normal);
            font-weight: var(--text-weight-normal);
            text-align: left;
            white-space: nowrap;
          }
        }

        &:last-child {
          border-bottom: none;
        }
      }
    }

    & > thead > tr:not(:last-child) > th,
    & > thead > tr:not(:last-child) > td,
    & > tbody > tr:not(:last-child) > th,
    & > tbody > tr:not(:last-child) > td,
    & > tfoot > tr:not(:last-child) > th,
    & > tfoot > tr:not(:last-child) > td,
    & > thead:not(:last-child),
    & > tbody:not(:last-child),
    & > tfoot:not(:last-child) {
      border-bottom: 1px solid var(--border-color-normal);
    }
  }

  .pagination {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: var(--spacing-gap-normal);
    color: var(--text-color-normal);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-normal);

    .pagination-buttons {
      display: flex;
      align-items: center;
      gap: var(--spacing-gap-normal);

      & button {
        border: none;
        background-color: transparent;
        cursor: pointer;
        font-size: var(--text-size-highlight);
        font-weight: var(--text-weight-highlight);
      }
    }

    & select {
      padding: var(--spacing-padding-small);
      border: 1px solid var(--border-color-lower);
      border-radius: var(--border-radius);
      margin-left: var(--spacing-gap-normal);
      background-color: var(--background-color-elevated-high);
      box-shadow: 0 1px 2px var(--background-color-selected);
      color: var(--text-color-highlight);
      cursor: pointer;
      font-size: var(--text-size-highlight);
      font-weight: var(--text-weight-highlight);
    }
  }
}
</style>
