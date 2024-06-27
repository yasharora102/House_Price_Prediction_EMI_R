library(shiny)
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)
library(lightgbm)
library(bslib)
library(shinydashboard)
library(plotly)

source("train.R")
# Clear the environment
rm(list = ls())

interest_rates <- read_csv("interest_rates.csv")
# Load models and data
load_models <- function() {
  rf_regressor <- readRDS("models/RF_model.rds")
  lr_regressor <- readRDS("models/lm_model.rds")
  dt_regressor <- readRDS("models/DT_model.rds")
  xgb_regressor <- readRDS("models/XGB_model.rds")
  lgb_regressor <- readRDS("models/LGB_model.rds")
  
  list(rf_regressor, lr_regressor, dt_regressor, xgb_regressor, lgb_regressor)
}

models <- load_models()
rf_regressor <- models[[1]]
lr_regressor <- models[[2]]
dt_regressor <- models[[3]]
xgb_regressor <- models[[4]]
lgb_regressor <- models[[5]]

# Load dictionaries
city_dict <- read_csv("City_dict.csv")
area_dict <- read_csv("Area_dict.csv")

# Prediction function
model_prediction <- function(city, area, sqft, bhk, park, ac, wifi, lift, security) {
  city_index <- which(city_dict$City == city)
  area_index <- which(area_dict$Area == area)
  
  new_df <- data.frame(
    City = city_index,
    Area = area_index,
    total_sqft = sqft,
    BHK = bhk,
    Parking = park,
    AC = ac,
    Wifi = wifi,
    Lift = lift,
    Security = security
  )
  
  preds <- c(
    predict(rf_regressor, newdata = new_df),
    predict(lr_regressor, newdata = new_df),
    predict(dt_regressor, newdata = new_df),
    predict(xgb_regressor, newdata = as.matrix(new_df)),
    predict(lgb_regressor, newdata = as.matrix(new_df))
  )
  
  return(preds)
}

# EMI calculation function
emi_calc <- function(principal, rate, tenure_years) {
  tenure_months <- tenure_years * 12
  monthly_rate <- rate / (12 * 100)
  emi <- (principal * monthly_rate * (1 + monthly_rate)^tenure_months) / ((1 + monthly_rate)^tenure_months - 1)
  return(emi)
}

# Server
server <- function(input, output, session) {
  Original <- read_csv("files/Original.csv")
  
  output$location_ui <- renderUI({
    city <- input$city
    original_dataset <- filter(Original, City == city)
    all_locations <- sort(unique(original_dataset$Area))
    selectInput("location", "Location", choices = all_locations)
  })
  
  user_data <- reactive({
    data <- data.frame(
      City = input$city,
      Location = input$location,
      Sqft = input$sqft,
      BHK = input$bhk,
      Park = input$park,
      AC = input$ac,
      Wifi = input$wifi,
      Lift = input$lift,
      Security = input$security
    )
    return(data)
  })
  
  preds <- reactive({
    model_prediction(input$city, input$location, input$sqft, input$bhk, input$park, input$ac, input$wifi, input$lift, input$security)
  })
  
  selected_pred <- reactive({
    switch(input$model,
           "Linear Regression" = preds()[2],
           "Random Forest (Best)" = preds()[1],
           "XGBoost" = preds()[4],
           "LGBM" = preds()[5],
           "DecisionTreeRegressor" = preds()[3]
    )
  })
  
  output$input_table <- renderTable({
    user_data()
  })
  
  output$prediction_ui <- renderUI({
    tagList(
      tags$h1(paste0("₹ ", round(selected_pred(), 2), " Lakhs"))
    )
  })
  
  output$meme_ui <- renderUI({
    meme_flag <- input$show_meme
    curr <- selected_pred()
    if (meme_flag) {
      if (curr <= 75) {
        img_src <- "img/1.png"
      } else if (curr > 75 && curr <= 100) {
        img_src <- "img/2.jpg"
      } else if (curr > 100 && curr <= 150) {
        img_src <- "img/3.png"
      } else {
        img_src <- "img/4.png"
      }
      tags$img(src = img_src, width = 300)
    }
  })
  
  output$model_performance_plot <- renderPlotly({
    model_performance <- data.frame(
      Model = c("Random Forest (Best)", "Linear Regression", "DecisionTreeRegressor", "XGBoost", "LGBM"),
      Prediction = unlist(preds())
    )
    
    p <- ggplot(model_performance, aes(x = Model, y = Prediction, group = 1)) +
      geom_line(color = "#803EF5", size = 1.2, linetype = "solid") +
      geom_point(color = "#FF5733", size = 4, shape = 21, fill = "white") +
      labs(title = "Model Performance Comparison",
           x = "Model",
           y = "Predicted Value (in Lakhs)") +
      theme_minimal(base_size = 15) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        axis.title.x = element_text(size = 15, face = "bold"),
        axis.title.y = element_text(size = 15, face = "bold"),
        axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12)
      )
    
    ggplotly(p)
  })
  
  output$accuracy_plot <- renderPlot({
    stats <- read_csv("files/all.csv")
    ggplot(stats, aes(x = Model, y = Accuracy)) +
      geom_line(color = "#803EF5") +
      geom_point(color = "#803EF5") +
      theme_dark()
  })
  
  
  
  bank_choices <- interest_rates %>%
    mutate(label = paste(BANK, "-", min)) %>%
    pull(label, BANK)
  
  output$emi_output <- renderUI({
    bank_interest <- interest_rates$min[interest_rates$BANK == gsub(" -.*", "", input$bank)]
    principal <- as.numeric(selected_pred()) * 100000  # Assuming prediction in Lakhs
    
    if (length(bank_interest) > 0 && !is.na(bank_interest)) {
      rate <- as.numeric(sub("%", "", bank_interest))  # Convert interest rate from string to numeric
      downpayment <- input$downpayment_amount  # Assuming downpayment provided by user
      
      loan_amount <- principal - downpayment
      emi <- emi_calc(loan_amount, rate, input$loan_tenure)  # Use user-selected loan tenure
      total_interest <- emi * input$loan_tenure * 12 - loan_amount
      total_amount <- loan_amount + total_interest
      
      tagList(
        h5("Monthly EMI:"),
        tags$p(paste0("₹ ", round(emi, 2))),
        h5("Principal amount:"),
        tags$p(paste0("₹ ", round(principal, 2))),
        h5("Total interest:"),
        tags$p(paste0("₹ ", round(total_interest, 2))),
        h5("Total amount:"),
        tags$p(paste0("₹ ", round(total_amount, 2)))
      )
    } else {
      tagList(
        h4("Please select a valid bank.")
      )
    }
  })
}

# UI
ui <- page_sidebar(
  title = "House Price Prediction",
  sidebar = sidebar(
    selectInput("model", "Choose Model", choices = c("Linear Regression", "Random Forest (Best)", "XGBoost", "LGBM", "DecisionTreeRegressor"), selected = "Random Forest (Best)"),
    selectInput("city", "Choose City", choices = c("Banglore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"), selected = "Banglore"),
    uiOutput("location_ui"),
    sliderInput("sqft", "Sqft", min = 100, max = 10000, value = 1000),
    sliderInput("bhk", "BHK", min = 1, max = 10, value = 2),
    sliderInput("park", "Park", min = 0, max = 1, value = 1),
    sliderInput("ac", "AC", min = 0, max = 1, value = 1),
    sliderInput("wifi", "Wifi", min = 0, max = 1, value = 1),
    sliderInput("lift", "Lift", min = 0, max = 1, value = 1),
    sliderInput("security", "Security", min = 0, max = 1, value = 1),
    checkboxInput("show_meme", "Show me the meme", FALSE)
  ),
  navset_card_underline(
    title = "House Price Prediction Dashboard",
    nav_panel(
      "Dashboard",
      fluidRow(
        column(
          width = 6,
          fluidRow(
            column(
              width = 12,
              card(
                card_header("Prediction"),
                card_body(
                  uiOutput("prediction_ui"),
                  conditionalPanel(
                    condition = "input.show_meme",
                    uiOutput("meme_ui")
                  )
                )
              )
            )
          ),
          fluidRow(
            column(
              width = 12,
              card(
                card_header("Model Performance"),
                card_body(
                  plotlyOutput("model_performance_plot")
                )
              )
            )
          )
        ),
        column(
          width = 6,
          card(
            card_header("EMI Calculator"),
            card_body(
              selectInput("bank", "Select Bank", choices = paste(interest_rates$BANK, "-", interest_rates$min)),
              sliderInput("downpayment_amount", "Downpayment Amount (₹)", min = 0, max = 10000000, value = 1000000),
              sliderInput("loan_tenure", "Loan Tenure (years)", min = 1, max = 30, value = 20),  # Slider for loan tenure
              uiOutput("emi_output")
            )
          )
        )
      )
    ),
    nav_panel(
      "About",
      h3("About this application"),
      p("This application is designed to predict house prices in Indian metropolitan areas using various machine learning models. Select a city and input various parameters to see the predicted house prices from different models. Additionally, you can view the model performance and accuracy comparisons.")
    )
  )
)

shinyApp(ui = ui, server = server)
