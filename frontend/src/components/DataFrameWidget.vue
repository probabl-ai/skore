<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref, watch } from "vue";
import TextInput from "./TextInput.vue";

export interface DataFrameWidgetProps {
  columns: string[];
  data: any[][];
}
const props = defineProps<DataFrameWidgetProps>();

const rowPerPage = ref(10);
const currentPage = ref(0);
const search = defineModel<string>("search");

const rows = computed(() => {
  if (search.value !== undefined && search.value.length > 0) {
    const searchToken = search.value.toLowerCase();
    return props.data.filter((row) => {
      const text = row.join(" ").toLowerCase();
      return text.includes(searchToken);
    });
  }
  return props.data;
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

watch(props.data, () => {
  currentPage.value = 0;
});

watch(props.columns, () => {
  currentPage.value = 0;
});
</script>

<template>
  <TextInput v-model="search" icon="icon-magnifying-glass" placeholder="Search" />
  <Simplebar>
    <table>
      <thead>
        <tr>
          <th v-for="(name, index) in props.columns" :key="index">{{ name }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, index) in visibleRows" :key="index">
          <td v-for="(value, index) in row" :key="index">{{ value }}</td>
        </tr>
      </tbody>
    </table>
  </Simplebar>
  <div class="pagination" v-if="totalPages > 1 || rowPerPage == rows.length">
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
</template>

<style scoped>
table {
  width: 100%;
  border: 1px solid var(--border-color-normal);
  border-radius: var(--border-radius);
  margin-top: var(--spacing-gap-small);
  border-collapse: separate;
  border-spacing: 0;
  text-align: right;

  & thead {
    background-color: var(--background-color-elevated);
    color: var(--text-color-normal);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-normal);

    & tr {
      & th {
        padding: var(--spacing-padding-small);
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
</style>
